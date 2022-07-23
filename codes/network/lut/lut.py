import torch
import torch.nn.functional as torch_func
from engine.checkpoint.functional import get_model_state_dict, load_model_state_dict
from .generator.generator import Generator3DLUT, Generator3DLUTSupplement
from .classifier.classifier import ClassifierResnet
from codes.network.build import BUILD_NETWORK_REGISTRY


@BUILD_NETWORK_REGISTRY.register()
class Gen3DLutModel(torch.nn.Module):
    def __init__(self, cfg):
        super(Gen3DLutModel, self).__init__()
        self.device = cfg.MODEL.DEVICE
        dim = 33
        is_zero = True
        nums = 8
        class_arch = 'resnet18'
        use_in = True
        down_factor = cfg.INPUT.DOWN_FACTOR
        if cfg.MODEL.NETWORK.get('LUT', None) is not None:
            dim = cfg.MODEL.NETWORK.LUT.get('DIM', dim)
            is_zero = cfg.MODEL.NETWORK.LUT.get('SUPP_ZERO', is_zero)
            nums = cfg.MODEL.NETWORK.LUT.get('SUPP_NUMS', nums)
            class_arch = cfg.MODEL.NETWORK.LUT.get('CLASS_ARCH', class_arch)
            use_in = cfg.MODEL.NETWORK.LUT.get('USE_INSTANCE', use_in)

        nums_class = nums + 1
        self.down_factor = down_factor
        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)

        self.lut0 = Generator3DLUT(dim=dim, device=self.device)
        self.lut1 = Generator3DLUTSupplement(dim=dim, nums=nums, device=self.device, is_zero=is_zero)
        self.classifier = ClassifierResnet(num_classes=nums_class, class_arch=class_arch, use_in=use_in, device=self.device)
        return

    def forward(self, x):
        dx = x
        if self.down_factor > 1:
            dx = torch_func.interpolate(dx, scale_factor=1/self.down_factor, mode='bilinear')

        cls_pre = self.classifier(dx).squeeze()

        assert cls_pre.shape[-1] - 1 == len(self.lut1)

        if len(cls_pre.shape) == 1:
            cls_pre = cls_pre.unsqueeze(0)

        shape = cls_pre.shape[0], 1, 1, 1
        combine_a = cls_pre[:, 0].reshape(shape) * self.lut0(x)
        lut1 = self.lut1(x)
        for i, val in lut1.items():
            combine_a += cls_pre[:, i + 1].reshape(shape) * val

        weights_norm = torch.mean(cls_pre ** 2)
        return combine_a, weights_norm

    def parameters(self):
        parameters = self.lut1.parameters()
        parameters.insert(0, self.lut0)
        parameters.insert(0, self.classifier.parameters())
        return parameters

    def state_dict(self):
        state_dict = dict()
        state = self.lut1.state_dict(offset=1)
        state[0] = get_model_state_dict(self.lut0)
        state_dict['lut'] = state
        state_dict['cls'] = get_model_state_dict(self.classifier)
        return state_dict

    def load_state_dict(self, state_dict:dict):
        load_model_state_dict(self.classifier, state_dict['cls'])
        load_model_state_dict(self.lut0, state_dict['lut'][0])
        self.lut1.load_state_dict(state_dict['lut'], offset=1)
        return

    def enable_parallel(self):
        self.lut0 = torch.nn.parallel.DataParallel(self.lut0)
        self.lut1.enable_parallel()
        return

    def enable_model_distributed(self, gpu_id):
        lut0 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.lut0)
        self.lut0 = torch.nn.parallel.DistributedDataParallel(lut0, device_ids=[gpu_id])
        self.lut1.enable_model_distributed(gpu_id)
        return

    def train(self):
        self.lut0.train()
        self.lut1.train()
        return

    def eval(self):
        self.lut0.eval()
        self.lut1.eval()
        return

    def calc_tv(self, tv_model):
        tv0, mn0 = tv_model(self.lut0)
        tv1 = list()
        mn1 = list()
        for k, lut in self.lut1.foreach():
            t, m = tv_model(lut)
            tv1.append(t)
            mn1.append(m)

        tv = sum(tv1) + tv0
        mn = sum(mn1) + mn0
        return tv, mn
