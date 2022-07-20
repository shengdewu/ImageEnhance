import torch
import torch.nn.functional as torch_func
from .ted import TransformedEncoderDecoder
from ..curl_tool import CureApply
from codes.network.build import BUILD_NETWORK_REGISTRY
import logging

__all__ = ['CURLFullNet']


class CURLConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, is_down=True, kernel_size=3, stride=2, padding=1):
        super(CURLConvBlock, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = torch.nn.LeakyReLU()
        self.down = None
        if is_down:
            self.down = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        return

    def forward(self, x):
        out = self.relu(self.conv2d(x))
        if self.down is not None:
            out = self.down(out)
        return out


class CurveLayer(torch.nn.Module):
    def __init__(self, kernel_number=64, knot_points=48, device='cuda'):
        super(CurveLayer, self).__init__()
        self.rgb_layer1 = CURLConvBlock(kernel_number, kernel_number)
        self.rgb_layer2 = CURLConvBlock(kernel_number, kernel_number)
        self.rgb_layer3 = CURLConvBlock(kernel_number, kernel_number)
        self.rgb_layer4 = CURLConvBlock(kernel_number, kernel_number, is_down=False)
        self.rgb_avg = torch.nn.AdaptiveAvgPool2d(1)
        self.rgb_fc = torch.nn.Linear(kernel_number, knot_points)
        self.rgb_dropout = torch.nn.Dropout(0.5)

        self.device = device
        self.to(device)
        return

    def forward(self, x):
        x = self.rgb_layer1(x)
        x = self.rgb_layer2(x)
        x = self.rgb_layer3(x)
        x = self.rgb_layer4(x)
        x = self.rgb_avg(x)
        x = x.view(x.size()[0], -1)
        x = self.rgb_dropout(x)
        r = self.rgb_fc(x)
        return r


@BUILD_NETWORK_REGISTRY.register()
class CURLFullNet(torch.nn.Module):
    def __init__(self, cfg):
        super(CURLFullNet, self).__init__()

        self.down_factor = cfg.INPUT.DOWN_FACTOR
        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)
        kernel_number = 64
        knot_points = 48
        if cfg.MODEL.NETWORK.get('CURL_FULL_NET', None) is not None:
            kernel_number = cfg.MODEL.NETWORK.CURL_FULL_NET.get('KERNEL_NUMBER', kernel_number)
            knot_points = cfg.MODEL.NETWORK.CURL_FULL_NET.get('KNOT_POINTS', knot_points)

        assert knot_points % 3 == 0, 'the {} must be divisible by 3'.format(knot_points)

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create network {} \nkernel_number: {} \nknot_points:{} \ndown_factor: {}'.format(self.__class__,
                                                                                                                                      kernel_number,
                                                                                                                                      knot_points,
                                                                                                                                      self.down_factor))

        self.ted = TransformedEncoderDecoder(kernel_number)
        self.cure = CurveLayer(kernel_number, knot_points, cfg.MODEL.DEVICE)
        return

    def forward(self, x):
        dx = x
        if self.down_factor > 1:
            dx = torch_func.interpolate(dx, scale_factor=1/self.down_factor, mode='bilinear')

        feat = self.ted(dx)
        r = self.cure(feat)
        img, gradient = CureApply.adjust_rgb(x, r)
        return img, gradient

