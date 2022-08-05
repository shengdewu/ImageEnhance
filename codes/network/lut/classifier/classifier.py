import torch
from . import create_resnet


class ClassifierResnet(torch.nn.Module):
    def __init__(self, num_classes, class_arch='resnet18', use_in=True, device='cpu'):
        super(ClassifierResnet, self).__init__()
        kwargs = dict()
        kwargs['num_classes'] = num_classes
        if use_in:
            kwargs['norm_layer'] = torch.nn.InstanceNorm2d

        self.resnet = create_resnet.create_resnet(arch=class_arch, **kwargs)

        self.to(device)
        return

    def forward(self, x):
        return self.resnet(x)


class ConvBnReLu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bn=True):
        super(ConvBnReLu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = None
        if bn:
            self.bn = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = torch.nn.LeakyReLU()
        return

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.relu(x)


class Classifier(torch.nn.Module):
    def __init__(self, num_classes, use_in=True, device='cpu'):
        super(Classifier, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Upsample(size=(256, 256), mode='bilinear'),
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.InstanceNorm2d(16, affine=True),
            ConvBnReLu(16, 32),
            ConvBnReLu(32, 64),
            ConvBnReLu(64, 128),
            ConvBnReLu(128, 128, bn=False),
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(128, num_classes, 8, padding=0),
        )

        self.to(device)
        return

    def init_normal_classifier(self):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.InstanceNorm2d):
                if m.affine:
                    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)

        torch.nn.init.constant_(self.model[-1].bias.data, 1.0)
        return

    def forward(self, img_input):
        return self.model(img_input)
