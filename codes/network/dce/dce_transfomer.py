import torch
import torch.nn.functional as torch_func
import logging
from codes.network.build import BUILD_NETWORK_REGISTRY
from codes.network.transformer_unet import TransformerUnet


@BUILD_NETWORK_REGISTRY.register()
class DceTransformerNet(torch.nn.Module):
    def __init__(self, cfg):
        super(DceTransformerNet, self).__init__()
        depth = 3
        num_groups = 2
        features = 32
        num_bottleneck_blocks = 1
        self.cure_nums = 8
        if cfg.MODEL.NETWORK.get('DCE_NET', None) is not None:
            features = cfg.MODEL.NETWORK.DCE_NET.get('KERNEL_NUMBER', features)
            depth = cfg.MODEL.NETWORK.DCE_NET.get('DEPTH', depth)
            num_groups = cfg.MODEL.NETWORK.DCE_NET.get('GROUPS', num_groups)
            num_bottleneck_blocks = cfg.MODEL.NETWORK.DCE_NET.get('BOTTLENECK', num_bottleneck_blocks)
            self.cure_nums = cfg.MODEL.NETWORK.DCE_NET.get('CURE_NUMS', self.cure_nums)

        self.down_factor = cfg.INPUT.DOWN_FACTOR
        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create network {}:\n down_factor {}\nfeatures {}\n'
                                                    'depth {}\n num_groups {}\n num_bottleneck_blocks {}\n'
                                                    'cure_nums {}\n'.format(self.__class__, self.down_factor, features,
                                                                            depth, num_groups, num_bottleneck_blocks, self.cure_nums))

        self.up_stage = torch.nn.UpsamplingBilinear2d(scale_factor=self.down_factor)

        self.stem = torch.nn.Sequential(
            TransformerUnet(in_channels=3, out_channels=3, depth=depth, num_groups=num_groups, features=features, num_bottleneck_blocks=num_bottleneck_blocks),
            torch.nn.Tanh()
        )
        return

    def enhance(self, x, cure):
        for n in range(self.cure_nums):
            x = x + cure * (torch.pow(x, 2) - x)
        return x + cure * (torch.pow(x, 2) - x)

    def forward(self, x):
        dx = x
        if self.down_factor > 1:
            dx = torch_func.interpolate(x, scale_factor=1/self.down_factor, mode='bilinear')
        cure = self.stem(dx)
        if self.down_factor > 1:
            cure = self.up_stage(cure)
        return self.enhance(x, cure), cure
