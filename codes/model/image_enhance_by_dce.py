import logging
import torch
from codes.model.build import BUILD_MODEL_REGISTRY
from engine.loss.vgg_loss import PerceptualLoss
from .image_enhance_by_zero_dce import ZeroDceModel
from codes.lap_pyramid import LapPyramidConv


@BUILD_MODEL_REGISTRY.register()
class DceModel(ZeroDceModel):
    def __init__(self, cfg):
        super(DceModel, self).__init__(cfg)

        self.vgg_loss = PerceptualLoss(cfg.MODEL.VGG.LAYER, device=self.device, path=cfg.MODEL.VGG.PATH)
        self.pixel_loss = torch.nn.MSELoss().to(self.device)

        self.lambda_vgg = cfg.SOLVER.LOSS.LAMBDA_VGG
        self.lambda_pixel = cfg.SOLVER.LOSS.LAMBDA_PIXEL

        self.pyramid = None
        if cfg.INPUT.get('PYRAMID_LEVEL', 0) > 0:
            self.pyramid = LapPyramidConv(cfg.INPUT.PYRAMID_LEVEL, device=self.device)
            logging.getLogger(self.default_log_name).info('enable pyramid, level = {}'.format(cfg.INPUT.PYRAMID_LEVEL))
        return

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        """
        device_input = data['input'].to(self.device, non_blocking=True)
        device_gt = data['expert'].to(self.device, non_blocking=True)

        if self.pyramid is not None:
            input_lap, _ = self.pyramid.pyramid_decompose(device_input)
            _, ref_gauss = self.pyramid.pyramid_decompose(device_gt)
            device_input = input_lap[-1]
            device_gt = ref_gauss[-1]

        enhance_img, cure = self.g_model(device_input)

        spa_loss = torch.mean(self.spa_loss(device_input, enhance_img))
        col_loss = torch.mean(self.col_loss(enhance_img))
        exp_loss = self.exp_loss(enhance_img)
        tv_loss = self.tv_loss(cure)

        pixel_loss = self.pixel_loss(enhance_img, device_gt)
        vgg_loss = self.vgg_loss(enhance_img, device_gt)

        total_loss = self.lambda_spa * spa_loss + self.lambda_col * col_loss + \
                     self.lambda_exp * exp_loss + self.lambda_tv * tv_loss + \
                     self.lambda_pixel * pixel_loss + self.lambda_vgg * vgg_loss

        self.g_optimizer.zero_grad()
        total_loss.backward()
        self.g_optimizer.step()

        return {'total_loss': total_loss.item(),
                'spa_loss': spa_loss.item(),
                'col_loss': col_loss.item(),
                'exp_loss': exp_loss.item(),
                'tv_loss': tv_loss.item(),
                'pixel_loss': pixel_loss.item(),
                'vgg_loss': vgg_loss.item()}

    def generator(self, data):
        if self.pyramid is None:
            return self.g_model(data.to(self.device, non_blocking=True))

        input_data = data.to(self.device, non_blocking=True)
        input_lap, _ = self.pyramid_decom(input_data)
        low_res = self.g_model(input_lap[-1])
        input_lap[-1] = low_res
        return self.pyramid_recons(input_lap)
