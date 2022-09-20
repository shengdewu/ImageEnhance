import torch
import torch.nn.functional as tnf
import einops
import engine.model.depth_wise as emd


def conv3x3(in_channels, out_channels, bias=True, is_dw=True):
    if is_dw:
        return emd.DepthWiseSeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


def conv1x1(in_channels, out_channels, bias=True):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)


def conv_trans(in_channels, out_channels, bias=True):
    return torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=bias)


def conv_down(in_channels, out_channels, bias=True):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=bias)


def to_nhwc(x):
    return einops.rearrange(x, 'n c h w -> n h w c')


def to_nchw(x):
    return einops.rearrange(x, 'n h w c -> n c h w')


class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, device=None, dtype=None):
        super(LayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine, device, dtype)
        return

    def forward(self, x):
        x = to_nhwc(x)
        x = self.layer_norm(x)
        return to_nchw(x)


class UpSample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, use_bias=True):
        super(UpSample, self).__init__()
        self.conv = conv1x1(in_dim, out_dim, bias=use_bias)
        return

    def forward(self, x, size):
        """
        size = h, w
        """
        x = tnf.interpolate(x, size=size, mode='bilinear', align_corners=True)
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

        self.layer_norm = LayerNorm(in_dim)
        return

    def forward(self, x):
        short_cut = x
        x = self.layer_norm(x)
        return short_cut + self.block(x)


class ResidualDenseChannelAttentionBlock(torch.nn.Module):
    """Residual dense channel attention block. Used in Bottlenecks."""

    def __init__(self, in_channel, features, reduction=16, bias=True, dropout_rate=0.0):
        super(ResidualDenseChannelAttentionBlock, self).__init__()
        self.layer_normal = LayerNorm(in_channel)
        self.mlp = MLPBlock(in_dim=in_channel,
                            mlp_dim=features,
                            dropout=dropout_rate,
                            use_bias=bias)
        self.cal = ChannelAttentionLayer(in_dim=in_channel,
                                         out_dim=features,
                                         reduction=reduction,
                                         use_bias=bias)
        return

    def forward(self, x, deterministic=True):
        # y = torch.nn.LayerNorm([x.shape[1], x.shape[2], x.shape[3]])(x)
        y = self.layer_normal(x)
        y = self.mlp(y)
        y = self.cal(y)
        x = x + y
        return x


class BottleneckBlock(torch.nn.Module):
    def __init__(self, in_channel,
                 features,
                 num_groups=1,
                 channels_reduction=4,
                 dropout_rate=0.0,
                 bias=True):
        super().__init__()

        self.conv1 = conv1x1(in_channel, features, bias=bias)
        for idx in range(num_groups):
            rdcab = ResidualDenseChannelAttentionBlock(
                in_channel=in_channel,
                features=features,
                reduction=channels_reduction,
                bias=bias,
                dropout_rate=dropout_rate)
            setattr(self, 'rdcab_{}'.format(idx), rdcab)

        self.num_groups = num_groups
        return

    def forward(self, x):
        """Applies the Mixer block to inputs."""
        assert x.ndim == 4  # Input has shape [batch, c,h, w]
        # input projection
        x = self.conv1(x)
        shortcut_long = x

        for i in range(self.num_groups):
            # Channel-mixing part, which provides within-patch communication.
            x = getattr(self, 'rdcab_{}'.format(i))(x)
        # long skip-connect
        x = x + shortcut_long
        return x


class UNetEncoderBlock(torch.nn.Module):
    def __init__(self, in_channel,
                 skip_channel,
                 features,
                 num_groups=1,
                 relu_slope=0.2,
                 channels_reduction=4,
                 down_sample=True,
                 bias=True):
        super().__init__()
        self.conv1 = conv1x1(in_channel + skip_channel, features, bias=bias)
        for idx in range(num_groups):
            rcab = ResidualChannelAttentionBlock(
                in_dim=features,
                reduction=channels_reduction,
                relu_slope=relu_slope,
                use_bias=bias)
            setattr(self, 'rcab_{}'.format(idx), rcab)

        self.num_groups = num_groups
        if down_sample:
            cd = conv_down(features, features, bias=bias)
            setattr(self, 'conv_down', cd)
        return

    def forward(self, x, skip=None):
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = tnf.interpolate(x, size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        shortcut_long = x
        for i in range(self.num_groups):
            x = getattr(self, 'rcab_{}'.format(i))(x)
        x = x + shortcut_long

        cd = getattr(self, 'conv_down', None)
        if cd is not None:
            return cd(x), x
        return x


class UNetDecoderBlock(torch.nn.Module):

    def __init__(self, in_channel,
                 skip_channel,
                 features,
                 num_groups=1,
                 relu_slope=0.2,
                 channels_reduction=4,
                 bias=True):
        super().__init__()

        self.conv_up = conv_trans(in_channel, features, bias=bias)
        self.encoder = UNetEncoderBlock(
            in_channel=features,
            skip_channel=skip_channel,
            features=features,
            num_groups=num_groups,
            relu_slope=relu_slope,
            channels_reduction=channels_reduction,
            down_sample=False,
            bias=bias)

    def forward(self, x, bridge=None):
        x = self.conv_up(x)  # self.features
        x = self.encoder(x, skip=bridge)
        return x


class TransformerCrossGateUnet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, depth=3, num_groups=2, features=32, num_bottleneck_blocks=2):
        super(TransformerCrossGateUnet, self).__init__()

        self.depth = depth
        channels_reduction = 4
        self.num_bottleneck_blocks = num_bottleneck_blocks
        use_bias = True

        self.input = conv3x3(in_channels=in_channels, out_channels=features, bias=use_bias)

        for i in range(self.depth):
            encoder = UNetEncoderBlock(
                in_channel=(2 ** i) * features,
                skip_channel=0,
                features=(2 ** (i+1)) * features,
                num_groups=num_groups,
                down_sample=True,
                relu_slope=0.2,
                channels_reduction=channels_reduction,
                bias=use_bias)
            setattr(self, 'encoder_{}'.format(i), encoder)

        for i in range(self.num_bottleneck_blocks):
            neck = BottleneckBlock(
                in_channel=(2 ** self.depth) * features,
                features=(2 ** self.depth) * features,
                num_groups=num_groups,
                channels_reduction=channels_reduction,
                dropout_rate=0.,
                bias=use_bias)
            setattr(self, 'bottleneck_{}'.format(i), neck)

        for i in reversed(range(self.depth)):
            for j in range(self.depth):
                if i == j:
                    continue
                up_sample = UpSample(in_dim=(2 ** (j+1)) * features, out_dim=(2 ** (i+1)) * features, use_bias=use_bias)
                setattr(self, 'cross_gate_up_sample_{}_{}'.format(i, j), up_sample)

            cross_gate = torch.nn.Sequential(
                conv1x1((2 ** (i+1)) * features * self.depth, (2 ** i) * features, bias=use_bias),
                conv3x3((2 ** i) * features, (2 ** i) * features, bias=use_bias)
            )
            setattr(self, 'cross_gate_{}'.format(i), cross_gate)

        for i in reversed(range(self.depth)):
            for j in range(self.depth):
                if i == j:
                    continue
                up_sample = UpSample(in_dim=(2 ** j) * features, out_dim=(2 ** i) * features, use_bias=use_bias)
                setattr(self, 'decoder_up_sample_{}_{}'.format(i, j), up_sample)

            decoder = UNetDecoderBlock(
                in_channel=(2 ** (i+1)) * features,
                skip_channel=(2 ** i) * features * self.depth,
                features=(2 ** i) * features,
                num_groups=num_groups,
                relu_slope=0.2,
                channels_reduction=channels_reduction,
                bias=use_bias)
            setattr(self, 'decoder_{}'.format(i), decoder)

        self.output = conv3x3(features, out_channels, bias=use_bias)

        return

    def forward(self, x):
        x = self.input(x)

        features = list()
        for i in range(self.depth):
            x, bridge = getattr(self, 'encoder_{}'.format(i))(x)
            features.append(bridge)

        for i in range(self.num_bottleneck_blocks):
            x = getattr(self, 'bottleneck_{}'.format(i))(x)

        signals = dict()
        for i in reversed(range(self.depth)):
            signal = list()
            for j in range(self.depth):
                if i == j:
                    signal.append(features[j])
                    continue
                signal.append(getattr(self, 'cross_gate_up_sample_{}_{}'.format(i, j))(features[j], size=features[i].shape[2:]))

            signals[i] = getattr(self, 'cross_gate_{}'.format(i))(torch.cat(signal, dim=1))

        for i in reversed(range(self.depth)):
            skip = list()
            for j in range(self.depth):
                if i == j:
                    skip.append(signals[j])
                    continue
                skip.append(getattr(self, 'decoder_up_sample_{}_{}'.format(i, j))(signals[j], size=signals[i].shape[2:]))
            x = getattr(self, 'decoder_{}'.format(i))(x, torch.concat(skip, dim=1))

        return self.output(x)


class TransformerUnet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, depth=3, num_groups=2, features=32, num_bottleneck_blocks=2):
        super(TransformerUnet, self).__init__()

        self.depth = depth
        channels_reduction = 4
        self.num_bottleneck_blocks = num_bottleneck_blocks
        use_bias = True

        self.input = conv3x3(in_channels=in_channels, out_channels=features, bias=use_bias)

        for i in range(self.depth):
            encoder = UNetEncoderBlock(
                in_channel=(2 ** i) * features,
                skip_channel=0,
                features=(2 ** (i+1)) * features,
                num_groups=num_groups,
                down_sample=True,
                relu_slope=0.2,
                channels_reduction=channels_reduction,
                bias=use_bias)
            setattr(self, 'encoder_{}'.format(i), encoder)

        for i in range(self.num_bottleneck_blocks):
            neck = BottleneckBlock(
                in_channel=(2 ** self.depth) * features,
                features=(2 ** self.depth) * features,
                num_groups=num_groups,
                channels_reduction=channels_reduction,
                dropout_rate=0.,
                bias=use_bias)
            setattr(self, 'bottleneck_{}'.format(i), neck)

        for i in reversed(range(self.depth)):
            decoder = UNetDecoderBlock(
                in_channel=(2 ** (i+1)) * features,
                skip_channel=(2 ** (i+1)) * features,
                features=(2 ** i) * features,
                num_groups=num_groups,
                relu_slope=0.2,
                channels_reduction=channels_reduction,
                bias=use_bias)
            setattr(self, 'decoder_{}'.format(i), decoder)

        self.output = conv3x3(features, out_channels, bias=use_bias)

        return

    def forward(self, x):
        x = self.input(x)

        features = list()
        for i in range(self.depth):
            x, bridge = getattr(self, 'encoder_{}'.format(i))(x)
            features.append(bridge)

        for i in range(self.num_bottleneck_blocks):
            x = getattr(self, 'bottleneck_{}'.format(i))(x)

        for i in reversed(range(self.depth)):
            x = getattr(self, 'decoder_{}'.format(i))(x, features[i])

        return self.output(x)


if __name__ == '__main__':
    """
    depth=3, features=32, num_bottleneck_blocks=1, num_groups=1 模型大小=17M
    depth=3, features=16, num_bottleneck_blocks=1, num_groups=2 模型大小=6.3M
    depth=3, features=32, num_bottleneck_blocks=1, num_groups=2 模型大小=25M
    depth=4, features=16, num_bottleneck_blocks=1, num_groups=2 模型大小=26M
    depth=4, features=32, num_bottleneck_blocks=1, num_groups=2 模型大小=102M
    depth=4, features=32, num_bottleneck_blocks=1, num_groups=1 模型大小=69M
    depth=4, features=32, num_bottleneck_blocks=2, num_groups=2 模型大小=108M
    """
    model = TransformerUnet(in_channels=3, out_channels=3, depth=4, features=32, num_bottleneck_blocks=1, num_groups=2)
    ones = torch.ones([2, 3, 512, 510], dtype=torch.float32)
    x = model(ones)
    torch.save(model, '1.pth')