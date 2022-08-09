import torch
import torch.nn.functional as tnf
from codes.network.build import BUILD_NETWORK_REGISTRY
import logging


__all__ = [
    'LaplacianPyramid'
]


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, use_in=False):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.bn1 = None
        if use_in:
            self.bn1 = torch.nn.InstanceNorm2d(out_channel)
        self.ac1 = torch.nn.LeakyReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn2 = None
        if use_in:
            self.bn2 = torch.nn.InstanceNorm2d(out_channel)
        self.ac2 = torch.nn.LeakyReLU(inplace=True)
        return

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.ac1(x)

        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.ac2(x)
        return x


class UpSample(torch.nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True, use_in=False):
        super().__init__()
        if bilinear:
            self.up = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            )
        else:
            self.up = torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

        self.bn = None
        if use_in:
            self.bn = torch.nn.InstanceNorm2d(out_channel)

        self.ac = torch.nn.LeakyReLU(inplace=True)
        return

    def forward(self, x, up_h, up_w):
        x = self.up(x)
        if self.bn is not None:
            x = self.bn(x)
        diff_h = up_h - x.shape[2]
        diff_w = up_w - x.shape[3]
        x = self.ac(x)
        return tnf.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                           diff_h // 2, diff_h - diff_h // 2])


class UnetBackBone(torch.nn.Module):
    def __init__(self, kernel_number=24, layers=4, use_in=False, max_channel=512, activation=None):
        super(UnetBackBone, self).__init__()
        self.layers = layers

        self.head = ConvBlock(3, kernel_number)

        in_channel = kernel_number
        channel_tuple = list()
        for i in range(layers):
            block_use_in = use_in
            if i >= layers-1:
                block_use_in = False

            in_channel = in_channel
            out_channel = min(in_channel * 2, max_channel)
            channel_tuple.append((in_channel, out_channel))

            down_block = torch.nn.Sequential(torch.nn.MaxPool2d(2),
                                             ConvBlock(in_channel, out_channel, block_use_in))
            setattr(self, 'down_block{}'.format(i), down_block)
            in_channel = min(in_channel * 2, max_channel)

        for i in range(layers-1, -1, -1):
            out_channel, in_channel = channel_tuple[i]
            conv_block = ConvBlock(out_channel*2, out_channel, use_in)
            setattr(self, 'conv_block{}'.format(i), conv_block)
            up_block = UpSample(in_channel, out_channel, use_in=use_in)
            setattr(self, 'up_block{}'.format(i), up_block)

        in_channel, _ = channel_tuple[0]
        self.final_conv = torch.nn.Conv2d(in_channel, 3, 1, 1, 0)
        self.activate = activation
        return

    def forward(self, x):
        head = self.head(x)

        skip_layer = [None for i in range(self.layers+1)]
        skip_layer[0] = head
        for i in range(0, self.layers):
            down_block = getattr(self, 'down_block{}'.format(i))
            skip_layer[i+1] = down_block(skip_layer[i])

        out = skip_layer[self.layers]
        for i in range(self.layers-1, -1, -1):
            conv_block = getattr(self, 'conv_block{}'.format(i))
            up_block = getattr(self, 'up_block{}'.format(i))
            up_in = torch.cat([up_block(out, skip_layer[i].shape[2], skip_layer[i].shape[3]), skip_layer[i]], 1)
            out = conv_block(up_in)

        if self.activate is not None:
            return self.activate(self.final_conv(out))
        return self.final_conv(out)


@BUILD_NETWORK_REGISTRY.register()
class LaplacianPyramid(torch.nn.Module):
    def __init__(self, cfg):
        super(LaplacianPyramid, self).__init__()
        kernel_number = 24
        layers = 4
        use_in = False
        max_channel = 512
        if cfg.MODEL.NETWORK.get('LAP_PYRAMID', None) is not None:
            kernel_number = cfg.MODEL.NETWORK.LAP_PYRAMID.get('KERNEL_NUMBER', kernel_number)
            layers = cfg.MODEL.NETWORK.LAP_PYRAMID.get('LAYERS', layers)
            use_in = cfg.MODEL.NETWORK.LAP_PYRAMID.get('USE_IN', use_in)
            max_channel = cfg.MODEL.NETWORK.LAP_PYRAMID.get('MAX_CHANNEL', max_channel)

        active = None
        if cfg.INPUT.get('DATA_MEAN', None) is not None and cfg.INPUT.get('DATA_STD', None) is not None:
            active = torch.nn.Tanh()

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create network {}\nkernel_number: {}\nlayers: {}\n'
                                                    'use_in: {}\nmax_channel: {}\nACTIVATE: {}'.format(self.__class__, kernel_number, layers, use_in, max_channel, active))

        self.sub_net = UnetBackBone(kernel_number, layers, use_in, max_channel=max_channel, activation=active)
        return

    def forward(self, x):
        return self.sub_net(x)
