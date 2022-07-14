import torch
import torch.nn.functional as torch_func
from engine.model.depth_wise import DepthWiseSeparableConv2d
from .curl_tool import CureApply
import logging
from codes.network.build import BUILD_NETWORK_REGISTRY


def conv3x3(in_planes, out_planes, stride=1, padding=1, use_dp=False, dilation=1):
    """3x3 convolution with padding"""
    if use_dp:
        return DepthWiseSeparableConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None, norm_layer=torch.nn.InstanceNorm2d, use_dp=False, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, use_dp=use_dp)
        self.bn1 = None
        if norm_layer is not None:
            self.bn1 = norm_layer(planes)
        self.relu = torch.nn.Tanh()
        self.conv2 = conv3x3(planes, planes, use_dp=use_dp)
        self.bn2 = None
        if norm_layer is not None:
            self.bn2 = norm_layer(planes)
        self.down_sample = down_sample
        self.stride = stride
        return

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


@BUILD_NETWORK_REGISTRY.register()
class CurlDownNet(torch.nn.Module):
    def __init__(self, cfg, is_rgb=True):
        super(CurlDownNet, self).__init__()

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create network {}'.format(self.__class__))

        kernel_number = 48
        knot_points = 30
        self.norm_layer = None
        use_dp = True
        if cfg.MODEL.NETWORK.get('CURL_NET', None) is not None:
            kernel_number = cfg.MODEL.NETWORK.CURL_NET.KERNEL_NUMBER #32
            knot_points = cfg.MODEL.NETWORK.CURL_NET.KNOT_POINTS  # 48
            if cfg.MODEL.NETWORK.CURL_NET.get('NORM_ENABLED', True):
                self.norm_layer = torch.nn.InstanceNorm2d
            use_dp = cfg.MODEL.NETWORK.CURL_NET.get('USE_DP', True)

        assert knot_points % 3 == 0, 'the {} must be divisible by 3'.format(knot_points)

        self.down_factor = cfg.INPUT.DOWN_FACTOR
        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)

        self.in_planes = kernel_number

        self.device = cfg.MODEL.DEVICE

        if self.norm_layer is not None:
            self.head = torch.nn.Sequential(
                torch.nn.Conv2d(3, kernel_number, kernel_size=3, stride=1, padding=1, bias=False),
                self.norm_layer(kernel_number),
                torch.nn.Tanh(),
            )
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Conv2d(3, kernel_number, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.Tanh(),
            )

        self.layer1 = self._make_layer(BasicBlock, kernel_number, 2, stride=1, use_dp=use_dp)
        self.layer2 = self._make_layer(BasicBlock, kernel_number*2, 4, stride=2, use_dp=use_dp)
        self.layer3 = self._make_layer(BasicBlock, kernel_number*4, 3, stride=2, use_dp=use_dp)
        self.layer4 = self._make_layer(BasicBlock, kernel_number*8, 2, stride=2, use_dp=use_dp)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(kernel_number * 8 * 2 * 2, knot_points),
        )

        return

    def _make_layer(self, block, planes, blocks, stride=1, use_dp=False, dilation=1):
        down_sample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if self.norm_layer is not None:
                down_sample = torch.nn.Sequential(
                    conv1x1(self.in_planes, planes * block.expansion, stride),
                    self.norm_layer(planes * block.expansion),
                )
            else:
                down_sample = torch.nn.Sequential(
                    conv1x1(self.in_planes, planes * block.expansion, stride),
                )

        layers = list()
        layers.append(block(self.in_planes, planes, stride, down_sample, self.norm_layer, use_dp, dilation))

        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, norm_layer=self.norm_layer, use_dp=use_dp))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        dx = x
        if self.down_factor > 1:
            dx = torch_func.interpolate(x, scale_factor=1/self.down_factor, mode='bilinear')

        s = self.head(dx)
        s = self.layer1(s)
        s = self.layer2(s)
        s = self.layer3(s)
        s = self.layer4(s)

        s = self.avg_pool(s)
        s = torch.flatten(s, 1)
        s = self.classifier(s)

        img, gradient_regularization = CureApply.adjust_rgb(x, s)

        return img, gradient_regularization
