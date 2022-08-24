import torch
import torch.nn.functional as tnf
import einops
from codes.network.build import BUILD_NETWORK_REGISTRY
import logging

__all__ = [
    'MAXIMS1',
    'MAXIMS2',
    'MAXIMS3',
    'MAXIMM1',
    'MAXIMM2',
    'MAXIMM3',
]


def conv3x3(in_channels, out_channels, bias=True):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


def conv1x1(in_channels, out_channels, bias=True):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)


def conv_trans(in_channels, out_channels, bias=True):
    return torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, padding=2, bias=bias)


def conv_down(in_channels, out_channels, bias=True):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=bias)


def block_images_einops(x, patch_size):
    b, c, h, w = x.shape
    x = x.permute(0, 2, 3, 1)
    grid_height = h // patch_size[0]
    grid_width = w // patch_size[1]
    x = einops.rearrange(x, 'n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c',
                         gh=grid_height, gw=grid_width,
                         fh=patch_size[0], fw=patch_size[1])
    return x.permute(0, 3, 1, 2)


def unblock_images_einops(x, grid_size, patch_size):
    x = x.permute(0, 2, 3, 1)
    x = einops.rearrange(x, 'n (gh, gw) (fh fw) c -> n (gh fh) (gw fw) c',
                         gh=grid_size[0], gw=grid_size[1],
                         fh=patch_size[0], fw=patch_size[1])
    return x.permute(0, 3, 1, 2)


def to_nhwc(x):
    return einops.rearrange(x, 'n c h w -> h h w c')


def to_nchw(x):
    return einops.rearrange(x, 'n h w c -> n c h w')


class UpSample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, ratio, use_bias=True):
        super(UpSample, self).__init__()
        self.ratio = ratio
        self.conv = conv1x1(in_dim, out_dim, bias=use_bias)
        return

    def forward(self, x):
        n, c, h, w = x.shape
        x = tnf.interpolate(x, size=(int(h*self.ratio), int(w*self.ratio)), mode='bilinear', align_corners=True)
        return self.conv(x)


class MLPBlock(torch.nn.Module):
    def __init__(self, in_dim, mlp_dim, dropout=0.0, use_bias=True):
        super(MLPBlock, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, mlp_dim, bias=use_bias),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(mlp_dim, in_dim, bias=use_bias)
        )
        return

    def forward(self, x):
        x = to_nhwc(x)
        x = self.mlp(x)
        return to_nchw(x)


class ChannelAttentionLayer(torch.nn.Module):
    """
        Squeeze-and-excitation block for channel attention
        ref: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, in_dim, out_dim, reduction=4, use_bias=True):
        super(ChannelAttentionLayer, self).__init__()

        self.mlp = torch.nn.Sequential(
            conv1x1(in_dim, out_dim // reduction, bias=use_bias),
            torch.nn.ReLU(),
            conv1x1(out_dim // reduction, in_dim, bias=use_bias)
        )

        self.sigmoid = torch.nn.Sigmoid()
        return

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)
        y = self.mlp(y)
        return x * self.sigmoid(y)


class ResidualChannelAttentionBlock(torch.nn.Module):
    """
    Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer
    """
    def __init__(self, in_dim, reduction=4, relu_slope=0.2, use_bias=True):
        super(ResidualChannelAttentionBlock, self).__init__()

        self.block = torch.nn.Sequential(
            conv3x3(in_dim, in_dim, bias=use_bias),
            torch.nn.LeakyReLU(negative_slope=relu_slope),
            conv3x3(in_dim, in_dim, bias=use_bias),
            ChannelAttentionLayer(in_dim, in_dim, reduction, use_bias)
        )
        return

    def forward(self, x):
        short_cut = x
        n, c, h, w = x.shape
        x = torch.nn.LayerNorm([c, h, w])(x)
        return short_cut + self.block(x)


class SpatialGatingUnit(torch.nn.Module):
    """ Spatial Gating Unit

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, h_size, use_bias=True):
        super().__init__()
        self.proj = torch.nn.Linear(h_size, h_size, bias=use_bias)
        return

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        torch.nn.init.normal_(self.proj.weight, std=1e-6)
        torch.nn.init.ones_(self.proj.bias)
        return

    def forward(self, x):
        u, v = x.chunk(2, dim=1)
        v_b, v_c, v_h, v_w = v.shape
        v = torch.nn.LayerNorm([v_c, v_h, v_w])(v)
        v = self.proj(torch.swapaxes(v, -1, -2))
        v = torch.swapaxes(v, -1, -2)
        return u * (v + 1.)


class GridGlobalMixLayer(torch.nn.Module):
    """Grid gMLP layer that performs global mixing of tokens."""
    def __init__(self, in_channel, grid_size, bias=True, factor=2., dropout=0.):
        super(GridGlobalMixLayer, self).__init__()
        self.linear_1 = torch.nn.Linear(in_channel, in_channel*factor, bias=bias)
        self.linear_2 = torch.nn.Linear(in_channel, in_channel, bias=bias)
        self.gate = SpatialGatingUnit(grid_size[0]*grid_size[1])
        self.grid_size = grid_size
        self.dropout = dropout
        return

    def forward(self, x, deterministic=True):
        n, c, h, w = x.shape
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        x = block_images_einops(x, patch_size=(fh, fw))
        # gMLP1: Global (grid) mixing part, provides global grid communication.
        _n, _c, _h, _w = x.shape
        y = torch.nn.LayerNorm(_c, _h, _w)(x)

        y = torch.swapaxes(y, -1, -3)
        y = self.linear_1(y)
        y = torch.swapaxes(y, -1, -3)
        y = tnf.gelu(y)

        y = self.gate(y)

        y = torch.swapaxes(y, -1, -3)
        y = self.linear_2(y)
        y = torch.swapaxes(y, -1, -3)
        y = tnf.dropout(y, self.dropout, deterministic)

        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


class BlockGatingUnit(torch.nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.

    The 'spatial' dim is defined as the **second last**.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self, w_size, bias=True):
        super(BlockGatingUnit, self).__init__()
        self.linear = torch.nn.Linear(w_size, w_size, bias=bias)
        return

    def forward(self, x):
        u, v = x.chunk(2, dim=1)
        v = torch.nn.LayerNorm([v.shape[1], v.shape[2], v.shape[3]])(v)
        v = self.linear(v)
        return u * (v + 1.)


class BlockGatingMLPLayer(torch.nn.Module):
    """Block gMLP layer that performs local mixing of tokens."""
    def __init__(self, in_channel, block_size, bias=True, factor=2., dropout=0.):
        super(BlockGatingMLPLayer, self).__init__()
        self.linear_1 = torch.nn.Linear(in_channel, in_channel*factor, bias=bias)
        self.linear_2 = torch.nn.Linear(in_channel, in_channel, bias=bias)
        self.gating = BlockGatingUnit(block_size[0]*block_size[1])
        self.block_size = block_size
        self.dropout = dropout
        return

    def forward(self, x, deterministic=True):
        n, c, h, w = x.shape
        fh, fw = self.block_size
        gh, gw = h // fh, w // fw
        x = block_images_einops(x, patch_size=(fh, fw))
        # MLP2: Local (block) mixing part, provides within-block communication.
        y = torch.nn.LayerNorm([x.shape[1], x.shape[2], x.shape[3]])(x)
        y = torch.swapaxes(y, -1, -3)
        y = self.linear_1(y)
        y = torch.swapaxes(y, -1, -3)
        y = tnf.gelu(y)

        y = self.gating(y)
        y = torch.swapaxes(y, -1, -3)
        y = self.linear_2(y)
        y = torch.swapaxes(y, -1, -3)
        y = tnf.dropout(y, self.dropout, deterministic)
        x = x + y
        return unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))


class ResidualSplitHeadMultiAxisGMLPLayer(torch.nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, in_channel,
                 block_size,
                 grid_size,
                 block_gmlp_factor=2,
                 grid_gmlp_factor=2,
                 input_proj_factor=2,
                 bias=True,
                 dropout_rate=0.):
        super(ResidualSplitHeadMultiAxisGMLPLayer, self).__init__()
        self.linear1 = torch.nn.Linear(in_channel, in_channel * input_proj_factor, bias=bias)
        self.gridgmlpLayer = GridGlobalMixLayer(in_channel=in_channel,
                                                grid_size=grid_size,
                                                bias=bias,
                                                factor=grid_gmlp_factor,
                                                dropout=dropout_rate)
        self.blockgmlpLayer = BlockGatingMLPLayer(in_channel=in_channel,
                                                  block_size=block_size,
                                                  bias=bias,
                                                  factor=block_gmlp_factor,
                                                  dropout=dropout_rate)
        self.linear2 = torch.nn.Linear(in_channel * input_proj_factor, in_channel, bias=bias)
        return

    def forward(self, x, deterministic=True):
        shortcut = x
        n, num_channels, h, w, = x.shape
        x = torch.nn.LayerNorm([num_channels, h, w])(x)
        x = torch.swapaxes(x, -1, -3)
        x = self.linear1(x)
        x = torch.swapaxes(x, -1, -3)

        x = tnf.gelu(x)
        u, v = x.chunk(2, dim=1)
        # GridGMLPLayer
        u = self.gridgmlpLayer(u)

        # BlockGMLPLayer
        v = self.blockgmlpLayer(v)
        x = torch.cat([u, v], dim=1)
        x = torch.swapaxes(x, -1, -3)
        x = self.linear2(x)
        x = torch.swapaxes(x, -1, -3)
        x = tnf.dropout(x, self.dropout_rate, deterministic)
        x = x + shortcut
        return x


class RDCAB(torch.nn.Module):
    """Residual dense channel attention block. Used in Bottlenecks."""
    def __init__(self, in_channel,features,reduction=16,bias=True,dropout_rate=0.0):
        super(RDCAB, self).__init__()
        self.mlpb = MLPBlock(in_dim=in_channel,
                             mlp_dim=features,
                             dropout=dropout_rate,
                             use_bias=bias)
        self.cal = ChannelAttentionLayer(in_dim=in_channel,
                                         out_dim=features,
                                         reduction=reduction,
                                         use_bias=bias)
        return

    def forward(self, x, deterministic=True):
        y = torch.nn.LayerNorm([x.shape[1], x.shape[2], x.shape[3]])(x)
        y = self.mlp(y)
        y = self.cal(y)
        x = x + y
        return x


class BottleneckBlock(torch.nn.Module):
    """The bottleneck block consisting of multi-axis gMLP block and RDCAB."""

    def __init__(self, in_channel,
                 features,
                 grid_size,
                 block_size,
                 num_groups=1,
                 block_gmlp_factor=2,
                 grid_gmlp_factor=2,
                 input_proj_factor=2,
                 channels_reduction=4,
                 dropout_rate=0.0,
                 bias=True):
        super().__init__()

        self.conv1 = conv1x1(in_channel, features, bias=bias)
        for idx in range(num_groups):
            RSHMAG = ResidualSplitHeadMultiAxisGMLPLayer(
                in_channel=in_channel,
                grid_size=grid_size,
                block_size=block_size,
                block_gmlp_factor=block_gmlp_factor,
                grid_gmlp_factor=grid_gmlp_factor,
                input_proj_factor=input_proj_factor,
                bias=bias,
                dropout_rate=dropout_rate)
            setattr(self, f"RSHMAG_{idx}", RSHMAG)
            Rdcab = RDCAB(
                in_channel=in_channel,
                features=features,
                reduction=channels_reduction,
                bias=bias,
                dropout_rate=dropout_rate)
            setattr(self, f"Rdcab_{idx}", Rdcab)

            self.num_groups = num_groups
            return

    def forward(self, x):
        """Applies the Mixer block to inputs."""
        assert x.ndim == 4  # Input has shape [batch, c,h, w]
        # input projection
        x = self.conv1(x)
        shortcut_long = x

        for i in range(self.num_groups):
            RSHMAG = getattr(self, f"RSHMAG_{i}")
            x = RSHMAG(x)
            # Channel-mixing part, which provides within-patch communication.
            Rdcab = getattr(self, f"Rdcab_{i}")
            x = Rdcab(x)
        # long skip-connect
        x = x + shortcut_long
        return x


class UNetEncoderBlock(torch.nn.Module):
    """Encoder block in MAXIM."""

    def __init__(self, in_channel,
                 in_channel_bridge,
                 features,
                 block_size,
                 grid_size,
                 num_groups=1,
                 lrelu_slope=0.2,
                 block_gmlp_factor=2,
                 grid_gmlp_factor=2,
                 input_proj_factor=2,
                 channels_reduction=4,
                 dropout_rate=0.0,
                 downsample=True,
                 use_global_mlp=True,
                 bias=True,
                 use_cross_gating=False):
        super().__init__()
        self.conv1 = conv1x1(in_channel + in_channel_bridge, features, bias=bias)
        self.conv_down = conv_down(features, features, bias=bias)
        for idx in range(num_groups):
            if use_global_mlp:
                RSHMAG = ResidualSplitHeadMultiAxisGMLPLayer(
                    in_channel=features,
                    block_size=block_size,
                    grid_size=grid_size,
                    block_gmlp_factor=block_gmlp_factor,
                    grid_gmlp_factor=grid_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    bias=bias,
                    dropout_rate=dropout_rate)
                setattr(self, f"RSHMAG_{idx}", RSHMAG)
            Rcab = ResidualChannelAttentionBlock(
                in_dim=features,
                reduction=channels_reduction,
                relu_slope=lrelu_slope,
                use_bias=bias)
            setattr(self, f"Rcab_{idx}", Rcab)

        self.CGB = CrossGatingBlock(
            in_channel_x=features,
            in_channel_y=features,
            features=features,
            block_size=block_size,
            grid_size=grid_size,
            dropout_rate=dropout_rate,
            input_proj_factor=input_proj_factor,
            upsample_y=False,
            bias=bias)

        self.num_groups = num_groups
        self.use_global_mlp = use_global_mlp
        return

    def forward(self, x,
                skip=None,
                enc=None,
                dec=None,
                *,
                deterministic: bool = True):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        shortcut_long = x
        for i in range(self.num_groups):
            if self.use_global_mlp:
                RSHMAG = getattr(self, f"RSHMAG_{i}")
                x = RSHMAG(x)
            Rcab = getattr(self, f"Rcab_{i}")
            x = Rcab(x)
        x = x + shortcut_long

        if enc is not None and dec is not None:
            assert self.use_cross_gating
            x, _ = self.CGB(x, enc + dec)

        if self.downsample:
            x_down = self.conv_down(x)
            return x_down, x
        else:
            return x


class UNetDecoderBlock(torch.nn.Module):
    """Decoder block in MAXIM."""

    def __init__(self, in_channel,
                 in_channel_bridge,
                 features,
                 grid_size,
                 block_size,
                 num_groups=1,
                 lrelu_slope=0.2,
                 block_gmlp_factor=2,
                 grid_gmlp_factor=2,
                 input_proj_factor=2,
                 channels_reduction=4,
                 dropout_rate=0.0,
                 downsample=True,
                 use_global_mlp=True,
                 bias=True):
        super().__init__()

        self.convt_up = conv_trans(in_channel, features, bias=bias)
        self.UNEB = UNetEncoderBlock(
            in_channel=features,
            in_channel_bridge=in_channel_bridge,
            features=features,
            block_size=block_size,
            grid_size=grid_size,
            num_groups=num_groups,
            lrelu_slope=lrelu_slope,
            block_gmlp_factor=block_gmlp_factor,
            grid_gmlp_factor=grid_gmlp_factor,
            input_proj_factor=input_proj_factor,
            channels_reduction=channels_reduction,
            dropout_rate=dropout_rate,
            downsample=False,
            use_global_mlp=use_global_mlp,
            bias=bias)

    def forward(self, x,
                bridge=None,
                deterministic=True):
        x = self.convt_up(x)  # self.features
        x = self.UNEB(x, skip=bridge, deterministic=deterministic)
        return x


class GetSpatialGatingWeights(torch.nn.Module):
    """Get gating weights for cross-gating MLP block."""

    def __init__(self, in_channel,
                 block_size,
                 grid_size,
                 input_proj_factor=2,
                 dropout_rate=0.0,
                 bias=True):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_channel, in_channel * input_proj_factor, bias=bias)
        self.linear2 = torch.nn.Linear(grid_size[0] * grid_size[1], grid_size[0] * grid_size[1], bias=bias)
        self.linear3 = torch.nn.Linear(block_size[0] * block_size[1], block_size[0] * block_size[1], bias=bias)
        self.linear4 = torch.nn.Linear(in_channel * input_proj_factor, in_channel, bias=bias)
        self.grid_size = grid_size
        self.block_size = block_size
        self.dropout_rate = dropout_rate
        return

    def forward(self, x):
        n, num_channels, h, w = x.shape

        # input projection
        x = torch.nn.LayerNorm([num_channels, h, w])(x)
        x = torch.swapaxes(x, -1, -3)
        x = self.linear1(x)
        x = torch.swapaxes(x, -1, -3)
        x = tnf.gelu(x)
        u, v = x.chunk(2, dim=1)

        # Get grid MLP weights
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        u = block_images_einops(u, patch_size=(fh, fw))
        u = torch.swapaxes(u, -1, -2)
        u = self.linear2(u)
        u = torch.swapaxes(u, -1, -2)

        u = unblock_images_einops(u, grid_size=(gh, gw), patch_size=(fh, fw))
        # Get Block MLP weights
        fh, fw = self.block_size
        gh, gw = h // fh, w // fw

        v = block_images_einops(v, patch_size=(fh, fw))

        v = self.linear3(v)

        v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(fh, fw))
        x = torch.cat([u, v], dim=1)

        x = torch.swapaxes(x, -1, -3)
        x = self.linear4(x)
        x = torch.swapaxes(x, -1, -3)
        x = tnf.dropout(x, p=self.dropout_rate)
        return x


class CrossGatingBlock(torch.nn.Module):
    """Cross-gating MLP block."""

    def __init__(self, in_channel_x,
                 in_channel_y,
                 features,
                 grid_size,
                 block_size,
                 dropout_rate=0.0,
                 input_proj_factor=2,
                 upsample_y=True,
                 bias=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.upsample_y = upsample_y
        self.convt_up = conv_trans(in_channel_y, features, bias=bias)
        self.conv1_1 = conv1x1(in_channel_x, features, bias=bias)
        self.conv1_2 = conv1x1(features, features, bias=bias)
        self.linear1 = torch.nn.Linear(features, features, bias=bias)
        self.getspatialgatingweights1 = GetSpatialGatingWeights(
            in_channel=features,
            block_size=block_size,
            grid_size=grid_size,
            dropout_rate=dropout_rate,
            bias=bias)
        self.linear2 = torch.nn.Linear(features, features, bias=bias)
        self.getspatialgatingweights2 = GetSpatialGatingWeights(
            in_channel=features,
            block_size=block_size,
            grid_size=grid_size,
            dropout_rate=dropout_rate,
            bias=bias)
        self.linear3 = torch.nn.Linear(features, features, bias=bias)
        self.linear4 = torch.nn.Linear(features, features, bias=bias)
        return

    def forward(self, x, y):
        # Upscale Y signal, y is the gating signal.
        if self.upsample_y:
            y = self.convt_up(y)
        x = self.conv1_1(x)
        y = self.conv1_2(y)
        assert y.shape == x.shape  # self.features
        shortcut_x = x
        shortcut_y = y

        # Get gating weights from X
        x = torch.nn.LayerNorm([x.shape[1], x.shape[2], x.shape[3]])(x)
        x = torch.swapaxes(x, -1, -3)
        x = self.linear1(x)
        x = torch.swapaxes(x, -1, -3)
        x = tnf.gelu(x)
        gx = self.getspatialgatingweights1(x)

        # Get gating weights from Y
        y = torch.nn.LayerNorm([y.shape[1], y.shape[2], y.shape[3]])(y)
        y = torch.swapaxes(y, -1, -3)
        y = self.linear2(y)
        y = torch.swapaxes(y, -1, -3)
        y = tnf.gelu(y)
        gy = self.getspatialgatingweights2(y)

        # Apply cross gating: X = X * GY, Y = Y * GX
        y = y * gx
        y = torch.swapaxes(y, -1, -3)
        y = self.linear3(y)
        y = torch.swapaxes(y, -1, -3)
        y = tnf.dropout(y, p=self.dropout_rate)
        y = y + shortcut_y

        x = x * gy  # gating x using y
        x = torch.swapaxes(x, -1, -3)
        x = self.linear4(x)
        x = torch.swapaxes(x, -1, -3)
        x = tnf.dropout(x, p=self.dropout_rate)
        x = x + y + shortcut_x  # get all aggregated signals
        return x, y


class SAM(torch.nn.Module):
    """Supervised attention module for multi-stage training.

      Introduced by MPRNet [CVPR2021]: https://github.com/swz30/MPRNet
      """

    def __init__(self, in_channel, features, output_channels=3, bias=True):
        super().__init__()
        self.conv3_1 = conv3x3(in_channel, features, bias=bias)
        self.conv3_2 = conv3x3(features, output_channels, bias=bias)
        self.conv3_3 = conv3x3(output_channels, features, bias=bias)
        return

    def forward(self, x, x_image, *,
                train: bool):
        """Apply the SAM module to the input and features.

        Args:
          x: the output features from UNet decoder with shape (h, w, c)
          x_image: the input image with shape (h, w, 3)
          train: Whether it is training

        Returns:
          A tuple of tensors (x1, image) where (x1) is the sam features used for the
            next stage, and (image) is the output restored image at current stage.
        """
        # Get features
        x1 = self.conv3_1(x)
        # Output restored image X_s
        if self.output_channels == 3:
            image = self.conv3_2(x) + x_image
        else:
            image = self.conv3_2(x)
        # Get attention maps for features
        x2 = self.conv3_3(image)
        x2_fun = torch.nn.Sigmoid()
        x2 = x2_fun(x2)
        # Get attended feature maps
        x1 = x1 * x2
        # Residual connection
        x1 = x1 + x
        return x1, image


class MAXIM(torch.nn.Module):
    """The MAXIM model function with multi-stage and multi-scale supervision.

    For more model details, please check the CVPR paper:
    MAXIM: MUlti-Axis MLP for Image Processing (https://arxiv.org/abs/2201.02973)

    Attributes:
      features: initial hidden dimension for the input resolution.
      depth: the number of downsampling depth for the model.
      num_stages: how many stages to use. It will also affects the output list.
      num_groups: how many blocks each stage contains.
      bias: whether to use bias in all the conv/mlp layers.
      num_supervision_scales: the number of desired supervision scales.
      lrelu_slope: the negative slope parameter in leaky_relu layers.
      use_global_mlp: whether to use the multi-axis gated MLP block (MAB) in each
        layer.
      use_cross_gating: whether to use the cross-gating MLP block (CGB) in the
        skip connections and multi-stage feature fusion layers.
      high_res_stages: how many stages are specificied as high-res stages. The
        rest (depth - high_res_stages) are called low_res_stages.
      block_size_hr: the block_size parameter for high-res stages.
      block_size_lr: the block_size parameter for low-res stages.
      grid_size_hr: the grid_size parameter for high-res stages.
      grid_size_lr: the grid_size parameter for low-res stages.
      num_bottleneck_blocks: how many bottleneck blocks.
      block_gmlp_factor: the input projection factor for block_gMLP layers.
      grid_gmlp_factor: the input projection factor for grid_gMLP layers.
      input_proj_factor: the input projection factor for the MAB block.
      channels_reduction: the channel reduction factor for SE layer.
      num_outputs: the output channels.
      dropout_rate: Dropout rate.

    Returns:
      The output contains a list of arrays consisting of multi-stage multi-scale
      outputs. For example, if num_stages = num_supervision_scales = 3 (the
      model used in the paper), the output specs are: outputs =
      [[output_stage1_scale1, output_stage1_scale2, output_stage1_scale3],
       [output_stage2_scale1, output_stage2_scale2, output_stage2_scale3],
       [output_stage3_scale1, output_stage3_scale2, output_stage3_scale3],]
      The final output can be retrieved by outputs[-1][-1].
    """

    def __init__(self,
                 features=64,
                 depth=3,
                 num_stages=2,
                 num_groups=1,
                 use_bias=True,
                 num_supervision_scales=1,
                 lrelu_slope=0.2,
                 use_global_mlp=True,
                 use_cross_gating=True,
                 high_res_stages=2,
                 block_size_hr=(16, 16),
                 block_size_lr=(8, 8),
                 grid_size_hr=(16, 16),
                 grid_size_lr=(8, 8),
                 num_bottleneck_blocks=1,
                 block_gmlp_factor=2,
                 grid_gmlp_factor=2,
                 input_proj_factor=2,
                 channels_reduction=4,
                 num_outputs=3,
                 dropout_rate=0.0):
        super().__init__()
        self.num_supervision_scales = num_supervision_scales
        self.use_cross_gating = use_cross_gating
        self.num_stages = num_stages
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.depth = depth

        for idx_stage in range(num_stages):
            for i in range(num_supervision_scales):
                conv = conv3x3(
                    in_channels=3,
                    out_channels=(2 ** i) * features,
                    bias=use_bias)
                setattr(self, f"conv_{i}", conv)
                if idx_stage > 0:
                    if use_cross_gating:
                        block_size = block_size_hr if i < high_res_stages else block_size_lr
                        grid_size = grid_size_hr if i < high_res_stages else block_size_lr
                        CGB = CrossGatingBlock(
                            in_channel_x=features * (2 ** i),
                            in_channel_y=features * (2 ** i),
                            features=(2 ** i) * features,
                            block_size=block_size,
                            grid_size=grid_size,
                            dropout_rate=dropout_rate,
                            input_proj_factor=input_proj_factor,
                            upsample_y=False,
                            bias=use_bias)
                        setattr(self, f"CGB_{idx_stage}_{i}", CGB)
                    else:
                        if i == 0:
                            in_channel_tmp = 64
                        else:
                            in_channel_tmp = (2 ** i) * features * 2
                        _tmp = conv1x1(in_channel_tmp, (2 ** i) * features, bias=use_bias)
                        setattr(self, f"stage_{idx_stage}_input_catconv_{i}", _tmp)

            for i in range(depth):
                block_size = block_size_hr if i < high_res_stages else block_size_lr
                grid_size = grid_size_hr if i < high_res_stages else block_size_lr
                use_cross_gating_layer = True if idx_stage > 0 else False
                in_channel_skip = (2 ** i) * features if i < num_supervision_scales else 0
                if i == 0:
                    in_channel_temp = features
                else:
                    in_channel_temp = (2 ** (i - 1)) * features
                UEB = UNetEncoderBlock(
                    in_channel=in_channel_temp + in_channel_skip,
                    in_channel_bridge=0,
                    features=(2 ** i) * features,
                    num_groups=num_groups,
                    downsample=True,
                    lrelu_slope=lrelu_slope,
                    block_size=block_size,
                    grid_size=grid_size,
                    block_gmlp_factor=block_gmlp_factor,
                    grid_gmlp_factor=grid_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    channels_reduction=channels_reduction,
                    use_global_mlp=use_global_mlp,
                    dropout_rate=dropout_rate,
                    bias=use_bias,
                    use_cross_gating=use_cross_gating_layer)
                setattr(self, f"UEB{idx_stage}_{i}", UEB)

            for i in range(num_bottleneck_blocks):
                if i == 0:
                    in_channel_temp = (2 ** (depth - 1)) * features
                else:
                    in_channel_temp = (2 ** (depth - 1)) * features
                BLB = BottleneckBlock(
                    in_channel=in_channel_temp,
                    features=(2 ** (depth - 1)) * features,
                    block_size=block_size_lr,
                    grid_size=block_size_lr,
                    num_groups=num_groups,
                    block_gmlp_factor=block_gmlp_factor,
                    grid_gmlp_factor=grid_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    channels_reduction=channels_reduction,
                    dropout_rate=dropout_rate,
                    bias=use_bias)
                setattr(self, f"BLB_{idx_stage}_{i}", BLB)

            for i in reversed(range(depth)):
                # use larger blocksize at high-res stages
                block_size = block_size_hr if i < high_res_stages else block_size_lr
                grid_size = grid_size_hr if i < high_res_stages else block_size_lr

                # get additional multi-scale signals
                for j in range(depth):
                    in_channel_temp = (2 ** j) * features
                    _UpSampleRatio = UpSample(
                        in_dim=in_channel_temp,
                        out_dim=(2 ** i) * features,
                        ratio=2 ** (j - i),
                        use_bias=use_bias)
                    setattr(self, f"UpSampleRatio_{idx_stage}_{i}_{j}", _UpSampleRatio)

                # Use cross-gating to cross modulate features
                if use_cross_gating:
                    in_channel_x_temp = 384
                    in_channel_y_temp = 128
                    if i != depth - 1:
                        in_channel_x_temp = in_channel_x_temp // 2
                        in_channel_y_temp = (2 ** (i + 1)) * features
                    _CrossGatingBlock = CrossGatingBlock(
                        in_channel_x=in_channel_x_temp,
                        in_channel_y=in_channel_y_temp,
                        features=(2 ** i) * features,
                        block_size=block_size,
                        grid_size=grid_size,
                        dropout_rate=dropout_rate,
                        input_proj_factor=input_proj_factor,
                        upsample_y=True,
                        bias=use_bias)
                    setattr(self, f"stage_{idx_stage}_cross_gating_block_{i}", _CrossGatingBlock)
                else:
                    in_channel_x_temp = 384
                    in_channel_y_temp = 64
                    if i != depth - 1:
                        in_channel_x_temp = in_channel_x_temp // 2
                        in_channel_y_temp = (2 ** (i + 1)) * features
                    _tmp = conv1x1(in_channel_x_temp, (2 ** i) * features, bias=use_bias)
                    setattr(self, f"stage_{idx_stage}_cross_gating_block_no_use_cross_gating_conv11_{i}", _tmp)
                    _tmp = conv3x3((2 ** i) * features, (2 ** i) * features, bias=use_bias)
                    setattr(self, f"stage_{idx_stage}_cross_gating_block_no_use_cross_gating_conv33_{i}", _tmp)

            # start decoder. Multi-scale feature fusion of cross-gated features
            for i in reversed(range(depth)):
                # use larger blocksize at high-res stages
                block_size = block_size_hr if i < high_res_stages else block_size_lr
                grid_size = grid_size_hr if i < high_res_stages else block_size_lr

                in_channel_temp = 128
                for j in range(depth):
                    if j != 0:
                        in_channel_temp = in_channel_temp // 2
                    _UpSampleRatio = UpSample(
                        in_dim=in_channel_temp,
                        out_dim=(2 ** i) * features,
                        ratio=2 ** (depth - j - 1 - i),
                        use_bias=use_bias)
                    setattr(self, f"UpSampleRatio_skip_signals_{idx_stage}_{i}_{j}", _UpSampleRatio)
                in_channel_temp_UDB = 128
                in_channel_temp_UDB_skip = 384
                if i != depth - 1:
                    in_channel_temp_UDB = (2 ** (i + 1)) * features
                    in_channel_temp_UDB_skip = in_channel_temp_UDB_skip // 2
                _UNetDecoderBlock = UNetDecoderBlock(
                    in_channel=in_channel_temp_UDB,
                    in_channel_bridge=in_channel_temp_UDB_skip,
                    features=(2 ** i) * features,
                    grid_size=grid_size,
                    block_size=block_size,
                    num_groups=num_groups,
                    lrelu_slope=lrelu_slope,
                    block_gmlp_factor=block_gmlp_factor,
                    grid_gmlp_factor=grid_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    channels_reduction=channels_reduction,
                    dropout_rate=dropout_rate,
                    use_global_mlp=use_global_mlp,
                    bias=use_bias)
                setattr(self, f"stage_{idx_stage}_decoder_block_{i}", _UNetDecoderBlock)

                if i < num_supervision_scales:
                    if idx_stage < num_stages - 1:  # not last stage, apply SAM
                        _SAM = SAM(
                            in_channel=(2 ** i) * features,
                            features=(2 ** i) * features,
                            output_channels=num_outputs,
                            bias=use_bias)
                        setattr(self, f"stage_{idx_stage}_supervised_attention_module_{i}", _SAM)

                    else:  # Last stage, apply output convolutions
                        _Conv3x3 = conv3x3((2 ** i) * features, num_outputs, bias=use_bias)
                        setattr(self, f"stage_{idx_stage}_output_conv_{i}", _Conv3x3)

    def forward(self, x, *, train: bool = False):
        n, c, h, w, = x.shape  # input image shape
        shortcuts = []
        shortcuts.append(x)
        # Get multi-scale input images

        for i in range(1, self.num_supervision_scales):
            shortcuts.append(tnf.interpolate(x, (h // (2 ** i), w // (2 ** i))))
        outputs_all = []
        sam_features, encs_prev, decs_prev = [], [], []

        for idx_stage in range(self.num_stages):
            # Input convolution, get multi-scale input features
            x_scales = []
            for i in range(self.num_supervision_scales):
                conv = getattr(self, f"conv_{i}")
                x_scale = conv(shortcuts[i])
                # If later stages, fuse input features with SAM features from prev stage
                if idx_stage > 0:
                    # use larger blocksize at high-res stages
                    if self.use_cross_gating:
                        CGB = getattr(self, f"CGB_{idx_stage}_{i}")
                        x_scale, _ = CGB(x_scale, sam_features.pop())
                    else:
                        x_scale_temp = torch.cat([x_scale, sam_features.pop()], dim=1)
                        # print(i,x_scale_temp.shape,'x_scale_temp.shape')
                        x_scale = getattr(self, f"stage_{idx_stage}_input_catconv_{i}") \
                            (x_scale_temp)

                x_scales.append(x_scale)

            # start encoder blocks
            encs = []
            x = x_scales[0]  # First full-scale input feature
            for i in range(self.depth):
                # use larger blocksize at high-res stages, vice versa.
                use_cross_gating_layer = True if idx_stage > 0 else False
                # Multi-scale input if multi-scale supervision
                x_scale = x_scales[i] if i < self.num_supervision_scales else None

                # UNet Encoder block
                enc_prev = encs_prev.pop() if idx_stage > 0 else None
                dec_prev = decs_prev.pop() if idx_stage > 0 else None
                UEB = getattr(self, f"UEB{idx_stage}_{i}")
                x, bridge = UEB(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
                # Cache skip signals
                encs.append(bridge)
            # Global MLP bottleneck blocks
            for i in range(self.num_bottleneck_blocks):
                BLB = getattr(self, f"BLB_{idx_stage}_{i}")
                x = BLB(x)
            # cache global feature for cross-gating
            global_feature = x
            # start cross gating. Use multi-scale feature fusion
            skip_features = []

            for i in reversed(range(self.depth)):
                # use larger blocksize at high-res stages
                # get additional multi-scale signals
                signal = torch.cat([
                    getattr(self, f"UpSampleRatio_{idx_stage}_{i}_{j}") \
                        (enc) for j, enc in enumerate(encs)], dim=1)
                # Use cross-gating to cross modulate features
                if self.use_cross_gating:
                    skips, global_feature = getattr(self, f"stage_{idx_stage}_cross_gating_block_{i}") \
                        (signal, global_feature)
                else:

                    skips = getattr(self, f"stage_{idx_stage}_cross_gating_block_no_use_cross_gating_conv11_{i}") \
                        (signal)

                    skips = getattr(self, f"stage_{idx_stage}_cross_gating_block_no_use_cross_gating_conv33_{i}") \
                        (skips)
                skip_features.append(skips)
            # start decoder. Multi-scale feature fusion of cross-gated features
            outputs, decs, sam_features = [], [], []
            for i in reversed(range(self.depth)):
                # use larger blocksize at high-res stages
                # get multi-scale skip signals from cross-gating block

                signal = torch.cat([
                    getattr(self, f"UpSampleRatio_skip_signals_{idx_stage}_{i}_{j}")(skip)
                    for j, skip in enumerate(skip_features)], dim=1)
                # UNetDecoderBlock
                x = getattr(self, f"stage_{idx_stage}_decoder_block_{i}")(x, bridge=signal)
                # Cache decoder features for later-stage's usage
                decs.append(x)

                # output conv, if not final stage, use supervised-attention-block.
                if i < self.num_supervision_scales:
                    if idx_stage < self.num_stages - 1:  # not last stage, apply SAM
                        sam, output = getattr(self, f"stage_{idx_stage}_supervised_attention_module_{i}") \
                            (x, shortcuts[i], train=train)
                        outputs.append(output)
                        sam_features.append(sam)
                    else:  # Last stage, apply output convolutions
                        output = getattr(self, f"stage_{idx_stage}_output_conv_{i}")(x)
                        output = output + shortcuts[i]
                        outputs.append(output)
            # Cache encoder and decoder features for later-stage's usage
            encs_prev = encs[::-1]
            decs_prev = decs

            # Store outputs
            outputs_all.append(outputs)
        return outputs_all


@BUILD_NETWORK_REGISTRY.register()
class MAXIMS1(MAXIM):
    def __init__(self, cfg):
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('select {}'.format(self.__class__))
        cfg = {
            "features": 32,
            "depth": 3,
            "num_stages": 1,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        }

        super(MAXIMS1, self).__init__(**cfg)
        return


@BUILD_NETWORK_REGISTRY.register()
class MAXIMS2(MAXIM):
    def __init__(self, cfg):
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('select {}'.format(self.__class__))
        cfg = {
            "features": 32,
            "depth": 3,
            "num_stages": 2,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        }

        super(MAXIMS2, self).__init__(**cfg)
        return


@BUILD_NETWORK_REGISTRY.register()
class MAXIMS3(MAXIM):
    def __init__(self, cfg):
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('select {}'.format(self.__class__))
        cfg = {
            "features": 32,
            "depth": 3,
            "num_stages": 3,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        }

        super(MAXIMS3, self).__init__(**cfg)
        return


@BUILD_NETWORK_REGISTRY.register()
class MAXIMM1(MAXIM):
    def __init__(self, cfg):
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('select {}'.format(self.__class__))
        cfg = {
            "features": 64,
            "depth": 3,
            "num_stages": 1,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        }

        super(MAXIMM1, self).__init__(**cfg)
        return


@BUILD_NETWORK_REGISTRY.register()
class MAXIMM2(MAXIM):
    def __init__(self, cfg):
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('select {}'.format(self.__class__))
        cfg = {
            "features": 64,
            "depth": 3,
            "num_stages": 2,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        }

        super(MAXIMM2, self).__init__(**cfg)
        return


@BUILD_NETWORK_REGISTRY.register()
class MAXIMM3(MAXIM):
    def __init__(self, cfg):
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('select {}'.format(self.__class__))
        cfg = {
            "features": 64,
            "depth": 3,
            "num_stages": 3,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        }

        super(MAXIMM3, self).__init__(**cfg)
        return