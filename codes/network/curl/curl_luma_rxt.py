import torch
from typing import Type, Any, Callable, Union, List, Optional

import torchvision.models

from .curl_tool import CureApply
import torch.nn.functional as torch_func
import logging
from codes.network.build import BUILD_NETWORK_REGISTRY


__all__ = [
    'CurlLumaRXTNet'
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(torch.nn.Module):

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        down_sample: Optional[torch.nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        expansion: int = 4,
        active: torch.nn.Module = torch.nn.ReLU(inplace=True)
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.down_sample layers down_sample the input when stride != 1
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = torch.nn.InstanceNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = torch.nn.InstanceNorm2d(width)
        self.conv3 = conv1x1(width, planes * expansion)
        self.bn3 = torch.nn.InstanceNorm2d(planes * expansion)
        self.relu = active
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


@BUILD_NETWORK_REGISTRY.register()
class CurlLumaRXTNet(torch.nn.Module):
    def __init__(self, cfg):
        super(CurlLumaRXTNet, self).__init__()

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create network {}'.format(self.__class__))

        self.down_factor = cfg.INPUT.DOWN_FACTOR
        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)

        self.dilation = 1
        self.in_planes = cfg.MODEL.NETWORK.CURL_XT.get('IN_PLANES', 32)
        self.base_width = cfg.MODEL.NETWORK.CURL_XT.get('BASE_WIDTH', 4)
        self.cardinality = cfg.MODEL.NETWORK.CURL_XT.get('CARDINALITY', 16)
        self.expansion = cfg.MODEL.NETWORK.CURL_XT.get('EXPANSION', 1)
        knot_points = cfg.MODEL.NETWORK.CURL_XT.get('KNOT_POINTS', 15)
        self._norm_layer = torch.nn.InstanceNorm2d

        self.conv1 = torch.nn.Conv2d(1, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.InstanceNorm2d(self.in_planes)
        self.relu = torch.nn.LeakyReLU(inplace=True)

        in_planes = self.in_planes
        self.layer1 = self._make_layer(in_planes, 3, active=self.relu)
        self.layer2 = self._make_layer(in_planes*2, 4, stride=2, dilate=True, active=self.relu)
        self.layer3 = self._make_layer(in_planes*4, 6, stride=2, dilate=False, active=self.relu)
        self.layer4 = self._make_layer(in_planes*8, 3, stride=2, dilate=False, active=self.relu)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(self.in_planes, knot_points)
        return

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, dilate: bool = False, active: torch.nn.Module = torch.nn.ReLU(inplace=True)) -> torch.nn.Sequential:
        norm_layer = self._norm_layer
        down_sample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * self.expansion:
            down_sample = torch.nn.Sequential(
                conv1x1(self.in_planes, planes * self.expansion, stride),
                norm_layer(planes * self.expansion),
            )

        layers = list()
        layers.append(Bottleneck(self.in_planes, planes, stride, down_sample, self.cardinality, self.base_width, previous_dilation, expansion=self.expansion, active=active))
        self.in_planes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes, groups=self.cardinality, base_width=self.base_width, dilation=self.dilation, expansion=self.expansion, active=active))

        return torch.nn.Sequential(*layers)

    def forward(self, img, gray):
        d_gray = gray
        if self.down_factor > 1:
            d_gray = torch_func.interpolate(d_gray, scale_factor=1/self.down_factor, mode='bilinear')

        x = self.conv1(d_gray)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        img, gradient_regularization = CureApply.adjust_luma(img, gray, x)
        return img, gradient_regularization
