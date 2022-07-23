import torchvision.models


arch_block = {
    'resnet18': (torchvision.models.resnet18, torchvision.models.resnet.BasicBlock),
    'resnet34': (torchvision.models.resnet34, torchvision.models.resnet.BasicBlock),
    'resnet50': (torchvision.models.resnet50, torchvision.models.resnet.Bottleneck),
    'resnet101': (torchvision.models.resnet101, torchvision.models.resnet.Bottleneck),
    'resnet152': (torchvision.models.resnet152, torchvision.models.resnet.Bottleneck),
    'resnext50_32x4d': (torchvision.models.resnext50_32x4d, torchvision.models.resnet.Bottleneck),
    'resnext101_32x8d': (torchvision.models.resnext101_32x8d, torchvision.models.resnet.Bottleneck),
    'wide_resnet50_2': (torchvision.models.wide_resnet50_2, torchvision.models.resnet.Bottleneck),
    'wide_resnet101_2': (torchvision.models.wide_resnet101_2, torchvision.models.resnet.Bottleneck),
}


def create_resnet(arch='resnet152', **kwargs):
    return arch_block[arch][0](pretrained=False, **kwargs)
