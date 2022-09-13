import torch
import torch.nn.functional as torch_func
from .curl_tool import CureApply
import logging
from codes.network.build import BUILD_NETWORK_REGISTRY
from codes.network.transformer_unet import TransformerUnet


@BUILD_NETWORK_REGISTRY.register()
class CurlTransformerNet(torch.nn.Module):
    def __init__(self, cfg):
        super(CurlTransformerNet, self).__init__()

        depth = 3
        num_groups = 2
        features = 32
        num_bottleneck_blocks = 1
        knot_points = 36
        knot_channels = features * 2
        if cfg.MODEL.NETWORK.get('CURL_NET', None) is not None:
            features = cfg.MODEL.NETWORK.CURL_NET.get('KERNEL_NUMBER', features)
            depth = cfg.MODEL.NETWORK.CURL_NET.get('DEPTH', depth)
            num_groups = cfg.MODEL.NETWORK.CURL_NET.get('GROUPS', num_groups)
            num_bottleneck_blocks = cfg.MODEL.NETWORK.CURL_NET.get('BOTTLENECK', num_bottleneck_blocks)
            knot_points = cfg.MODEL.NETWORK.CURL_NET.get('KNOT_POINTS', knot_points)
            knot_channels = cfg.MODEL.NETWORK.CURL_NET.get('KNOT_CHANNELS', knot_channels)

        assert knot_points % 3 == 0, 'the {} must be divisible by 3'.format(knot_points)

        self.down_factor = cfg.INPUT.DOWN_FACTOR
        if cfg.INPUT.get('PYRAMID_LEVEL', 0) > 0:
            self.down_factor = 1
        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)

        self.device = cfg.MODEL.DEVICE

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create network {}:\n '
                                                    'down_factor {}\n'
                                                    'features {}\n'
                                                    'depth {}\n '
                                                    'num_groups {}\n '
                                                    'num_bottleneck_blocks {}\n'
                                                    'knot_points {}\n'
                                                    'knot_channels {} \n'.format(self.__class__,
                                                                              self.down_factor,
                                                                              features,
                                                                              depth,
                                                                              num_groups,
                                                                              num_bottleneck_blocks,
                                                                              knot_points,
                                                                              knot_channels))
        self.backbone = TransformerUnet(in_channels=3, out_channels=knot_channels, depth=depth, num_groups=num_groups, features=features, num_bottleneck_blocks=num_bottleneck_blocks)

        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(knot_channels, knot_points)
        self.dropout = torch.nn.Dropout(0.5)

        return

    def forward(self, x):
        dx = x
        if self.down_factor > 1:
            dx = torch_func.interpolate(x, scale_factor=1/self.down_factor, mode='bilinear')

        r = self.backbone(dx)
        r = self.avg(r)
        r = r.view(r.size()[0], -1)
        r = self.dropout(r)
        r = self.fc(r)

        rgb, gradient_regularization_rgb = CureApply.adjust_rgb(x, r)
        return rgb, gradient_regularization_rgb
