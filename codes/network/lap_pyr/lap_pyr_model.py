import torch
import torch.nn.functional as tnf
from codes.network.build import BUILD_NETWORK_REGISTRY
import logging


__all__ = [
    'LaplacianPyramid'
]


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.ac1 = torch.nn.LeakyReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.ac2 = torch.nn.LeakyReLU(inplace=True)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.ac2(x)
        return x


class UpSample(torch.nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            )
        else:
            self.up = torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.ac = torch.nn.LeakyReLU(inplace=True)
        return

    def forward(self, x, up_h, up_w):
        x = self.up(x)
        diff_h = up_h - x.shape[2]
        diff_w = up_w - x.shape[3]
        x = self.ac(x)
        return tnf.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                           diff_h // 2, diff_h - diff_h // 2])


class UnetBackBone(torch.nn.Module):
    def __init__(self, kernel_number=24, layers=4):
        super(UnetBackBone, self).__init__()
        self.max_pool = torch.nn.MaxPool2d(2)
        self.layers = layers

        self.head = ConvBlock(3, kernel_number)

        in_channel = kernel_number
        for i in range(layers):
            down_block = torch.nn.Sequential(torch.nn.MaxPool2d(2),
                                             ConvBlock(in_channel, in_channel*2))
            setattr(self, 'down_block{}'.format(i), down_block)
            in_channel = in_channel * 2

        self.bottle = ConvBlock(in_channel, in_channel * 2)
        in_channel = in_channel * 2

        for i in range(layers-1, -1, -1):
            conv_block = ConvBlock(in_channel, in_channel // 2)
            setattr(self, 'conv_block{}'.format(i), conv_block)
            up_block = UpSample(in_channel, in_channel // 2)
            setattr(self, 'up_block{}'.format(i), up_block)
            in_channel = in_channel // 2

        self.final_conv = torch.nn.Conv2d(in_channel, 3, 1, 1, 0)
        return

    def forward(self, x):
        head = self.head(x)

        skip_layer = [None for i in range(self.layers)]
        down_block = getattr(self, 'down_block{}'.format(0))
        skip_layer[0] = down_block(head)
        for i in range(1, self.layers-1):
            down_block = getattr(self, 'down_block{}'.format(0))
            skip_layer[i] = down_block(skip_layer[i-1])

        out = self.bottle(skip_layer[self.layers-1])

        for i in range(self.layers-1, 0, -1):
            conv_block = getattr(self, 'conv_block{}'.format(self.layers - 1))
            up_block = getattr(self, 'up_block{}'.format(self.layers - 1))
            up_in = torch.cat([up_block(out, skip_layer[i].shape[2], skip_layer[i].shape[3]), skip_layer[i]], 1)
            out = conv_block(up_in)

        return self.final_conv(out)


@BUILD_NETWORK_REGISTRY.register()
class LaplacianPyramid(torch.nn.Module):
    def __init__(self, cfg):
        super(LaplacianPyramid, self).__init__()
        kernel_number = 24
        layers = 4
        if cfg.MODEL.NETWORK.get('LAP_PYRAMID', None) is not None:
            kernel_number = cfg.MODEL.NETWORK.LAP_PYRAMID.get('KERNEL_NUMBER', kernel_number)
            layers = cfg.MODEL.NETWORK.LAP_PYRAMID.get('LAYERS', layers)

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create network {}\nkernel_number: {}\nlayers: {}'.format(self.__class__, kernel_number, layers))
        self.sub_net = UnetBackBone(kernel_number, layers)
        return

    def forward(self, x):
        return self.sub_net(x)
