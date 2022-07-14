import torch
import torch.nn.functional as torch_func
from engine.model.depth_wise import DepthWiseSeparableConv2d
from .curl_tool import CureApply
import logging
from codes.network.build import BUILD_NETWORK_REGISTRY


@BUILD_NETWORK_REGISTRY.register()
class CurlNet(torch.nn.Module):
    def __init__(self, cfg):
        super(CurlNet, self).__init__()

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create network {}'.format(self.__class__))

        kernel_number = 32
        knot_points = 48
        pre_knot_points = kernel_number
        if cfg.MODEL.NETWORK.get('CURL_NET', None) is not None:
            kernel_number = cfg.MODEL.NETWORK.CURL_NET.KERNEL_NUMBER #32
            knot_points = cfg.MODEL.NETWORK.CURL_NET.KNOT_POINTS  # 48
            pre_knot_points = cfg.MODEL.NETWORK.CURL_NET.get('PRE_KNOT_POINTS', pre_knot_points)

        assert knot_points % 3 == 0, 'the {} must be divisible by 3'.format(knot_points)

        self.down_factor = cfg.INPUT.DOWN_FACTOR
        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)

        self.device = cfg.MODEL.DEVICE

        self.stem_1 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(3, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.stem_2 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.stem_3 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.stem_4 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.stem_5 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number*2, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.stem_6 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number*2, kernel_number),
            torch.nn.LeakyReLU()
        )

        self.stem_7 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number*2, pre_knot_points),
            torch.nn.LeakyReLU()
        )

        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(pre_knot_points, knot_points)
        self.dropout = torch.nn.Dropout(0.5)

        return

    def forward(self, x):
        dx = x
        if self.down_factor > 1:
            dx = torch_func.interpolate(x, scale_factor=1/self.down_factor, mode='bilinear')
        s1 = self.stem_1(dx)
        s2 = self.stem_2(s1)
        s3 = self.stem_3(s2)
        s4 = self.stem_4(s3)
        s5 = self.stem_5(torch.cat([s3, s4], dim=1))
        s6 = self.stem_6(torch.cat([s2, s5], dim=1))
        r = self.stem_7(torch.cat([s1, s6], dim=1))
        r = self.avg(r)
        r = r.view(r.size()[0], -1)
        r = self.dropout(r)
        r = self.fc(r)

        rgb, gradient_regularization_rgb = CureApply.adjust_rgb(x, r)
        return rgb, gradient_regularization_rgb
