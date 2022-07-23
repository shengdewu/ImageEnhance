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


