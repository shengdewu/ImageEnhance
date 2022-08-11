import torch
import torch.nn.functional as tnf
from codes.discriminator.build import BUILD_DISCRIMINATOR_REGISTRY

__all__ = [
    'Discriminator',
    'MultiDiscriminator',
    'MultiDiscriminatorV2'
]


def discriminator_block_v1(in_filters, out_filters, normalization=False):
    layers = [torch.nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1), torch.nn.LeakyReLU(0.2)]
    if normalization:
        layers.append(torch.nn.InstanceNorm2d(out_filters, affine=True))
    return layers


def discriminator_block_v2(in_filters, out_filters, normalize=True):
    """Returns downsampling layers of each discriminator block"""
    layers = [torch.nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
    if normalize:
        layers.append(torch.nn.InstanceNorm2d(out_filters))
    layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
    return layers


@BUILD_DISCRIMINATOR_REGISTRY.register()
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.InstanceNorm2d(16, affine=True),
            *discriminator_block_v1(16, 32),
            *discriminator_block_v1(32, 64),
            *discriminator_block_v1(64, 128),
            *discriminator_block_v1(128, 128),
            torch.nn.Conv2d(128, 1, 8, padding=0)
        )

    def forward(self, img_input):
        return self.model(img_input)


@BUILD_DISCRIMINATOR_REGISTRY.register()
class MultiDiscriminator(torch.nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        # Extracts three discriminator models
        self.models = torch.nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                torch.nn.Sequential(
                    torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
                    torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
                    torch.nn.LeakyReLU(0.2),
                    torch.nn.InstanceNorm2d(16, affine=True),
                    *discriminator_block_v1(16, 32),
                    *discriminator_block_v1(32, 64),
                    *discriminator_block_v1(64, 128),
                    *discriminator_block_v1(128, 128),
                    torch.nn.Conv2d(128, 1, 8, padding=0)
                ),
            )

        self.down_sample = torch.nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


@BUILD_DISCRIMINATOR_REGISTRY.register()
class MultiDiscriminatorV2(torch.nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminatorV2, self).__init__()

        # Extracts three discriminator models
        self.models = torch.nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                torch.nn.Sequential(
                    *discriminator_block_v2(in_channels, 64, normalize=False),
                    *discriminator_block_v2(64, 128),
                    *discriminator_block_v2(128, 256),
                    *discriminator_block_v2(256, 512),
                    torch.nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.down_sample = torch.nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


@BUILD_DISCRIMINATOR_REGISTRY.register()
class MultiScaleExposureDis(torch.nn.Module):
    def __init__(self):
        super(MultiScaleExposureDis, self).__init__()

        def block(in_channels, out_channel, kernel_size, stride, padding, normalization=False):
            layers = [torch.nn.Conv2d(in_channels, out_channel, kernel_size, stride, padding)]
            if normalization:
                layers.append(torch.nn.BatchNorm2d(out_channel, affine=True))
            layers.append(torch.nn.LeakyReLU())
            return layers

        self.model = torch.nn.Sequential(
            *block(3, 8, 4, 2, 1),
            *block(8, 16, 4, 2, 1, True),
            *block(16, 32, 4, 2, 1),
            *block(32, 64, 4, 2, 1),
            *block(64, 128, 4, 2, 1),
            *block(128, 128, 4, 2, 1),
            *block(128, 256, 4, 2, 1),
            torch.nn.Conv2d(256, 1, 2, 2, 0)
        )
        return

    def forward(self, x):
        if x.shape[2] != 256 and x.shape[3] != 256:
            x = tnf.interpolate(x, (256, 256), mode='bilinear', align_corners=True)
        return self.model(x)

