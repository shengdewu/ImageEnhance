import torch
import torch.nn.functional as torch_func
from engine.model.depth_wise import DepthWiseSeparableConv2d
import logging
from codes.network.build import BUILD_NETWORK_REGISTRY


@BUILD_NETWORK_REGISTRY.register()
class DceNet(torch.nn.Module):
    def __init__(self, cfg):
        super(DceNet, self).__init__()
        kernel_number = 32
        if cfg.MODEL.NETWORK.get('DCE_NET', None) is not None:
            kernel_number = cfg.MODEL.NETWORK.DCE_NET.get('KERNEL_NUMBER', kernel_number)

        self.down_factor = cfg.INPUT.DOWN_FACTOR
        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create network {}'.format(self.__class__))

        self.up_stage = torch.nn.UpsamplingBilinear2d(scale_factor=self.down_factor)

        self.stem_1 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(3, kernel_number),
            torch.nn.ReLU()
        )

        self.stem_2 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number, kernel_number),
            torch.nn.ReLU()
        )

        self.stem_3 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number, kernel_number),
            torch.nn.ReLU()
        )

        self.stem_4 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number, kernel_number),
            torch.nn.ReLU()
        )

        self.stem_5 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number*2, kernel_number),
            torch.nn.ReLU()
        )

        self.stem_6 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number*2, kernel_number),
            torch.nn.ReLU()
        )

        self.stem_7 = torch.nn.Sequential(
            DepthWiseSeparableConv2d(kernel_number*2, 3),
            torch.nn.Tanh()
        )
        return

    def enhance(self, x, cure):
        x = x + cure * (torch.pow(x, 2) - x)
        x = x + cure * (torch.pow(x, 2) - x)
        x = x + cure * (torch.pow(x, 2) - x)
        enhance_img = x + cure * (torch.pow(x, 2) - x)
        x = enhance_img + cure * (torch.pow(enhance_img, 2) - enhance_img)
        x = x + cure * (torch.pow(x, 2) - x)
        x = x + cure * (torch.pow(x, 2) - x)
        return x + cure * (torch.pow(x, 2) - x)

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
        cure = self.stem_7(torch.cat([s1, s6], dim=1))
        if self.down_factor > 1:
            cure = self.up_stage(cure)
        return self.enhance(x, cure), cure
