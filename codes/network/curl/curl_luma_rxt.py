import torch
from typing import Type, Any, Callable, Union, List, Optional
from .curl_tool import CureApply
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
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[torch.nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        expansion: int = 4
    ) -> None:
        super(Bottleneck, self).__init__()
        width = base_width * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = torch.nn.InstanceNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = torch.nn.InstanceNorm2d(width)
        self.conv3 = conv1x1(width, planes * expansion)
        self.bn3 = torch.nn.InstanceNorm2d(planes * expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@BUILD_NETWORK_REGISTRY.register()
class CurlLumaRXTNet(torch.nn.Module):
    def __init__(self, cfg):
        super(CurlLumaRXTNet, self).__init__()

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create network {}'.format(self.__class__))

        self.dilation = 1
        self.in_planes = cfg.MODEL.NETWORK.CURL_XT.get('IN_PLANES', 64)
        self.base_width = cfg.MODEL.NETWORK.CURL_XT.get('BASE_WIDTH', 4)
        self.cardinality = cfg.MODEL.NETWORK.CURL_XT.get('CARDINALITY', 32)
        self.expansion = cfg.MODEL.NETWORK.CURL_XT.get('EXPANSION', 2)
        knot_points = cfg.MODEL.NETWORK.CURL_XT.get('KNOT_POINTS', 15)
        self._norm_layer = torch.nn.InstanceNorm2d

        self.conv1 = torch.nn.Conv2d(1, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.InstanceNorm2d(self.in_planes)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(self.in_planes, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = torch.nn.InstanceNorm2d(self.in_planes)

        self.layer1 = self._make_layer(self.in_planes, 3)
        self.layer2 = self._make_layer(self.in_planes, 4, stride=2, dilate=True)
        self.layer3 = self._make_layer(self.in_planes, 6, stride=2, dilate=False)
        self.layer4 = self._make_layer(self.in_planes, 3, stride=2, dilate=False)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(self.in_planes, knot_points)
        return

    def _make_layer(self, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> torch.nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * self.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.in_planes, planes * self.expansion, stride),
                norm_layer(planes * self.expansion),
            )

        layers = list()
        layers.append(Bottleneck(self.in_planes, planes, stride, downsample, self.cardinality, self.base_width, previous_dilation, expansion=self.expansion))
        self.in_planes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes, groups=self.cardinality, base_width=self.base_width, dilation=self.dilation, expansion=self.expansion))

        return torch.nn.Sequential(*layers)

    def forward(self, img, gray):
        x = self.conv1(gray)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        img, gradient_regularization = CureApply.adjust_luma(img, gray, x)
        return img, gradient_regularization
