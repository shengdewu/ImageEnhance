import torch
from codes.model.build import BUILD_MODEL_REGISTRY
from engine.loss.vgg_loss import PerceptualLoss
import engine.loss.ssim_loss as engine_ssim
from .image_enhance_pair_base import PairBaseModel
from codes.lap_pyramid import LapPyramidConv
import logging


@BUILD_MODEL_REGISTRY.register()
class CurlModel(PairBaseModel):
    def __init__(self, cfg):
        super(CurlModel, self).__init__(cfg)

        self.vgg_loss = PerceptualLoss(cfg.MODEL.VGG.LAYER, device=self.device, path=cfg.MODEL.VGG.PATH)
        self.ssim_loss = engine_ssim.SSIM()
        self.pixel_loss = torch.nn.L1Loss().to(self.device)
        self.cos_loss = torch.nn.CosineSimilarity(dim=1)

        self.lambda_ssim = cfg.SOLVER.LOSS.LAMBDA_SSIM
        self.lambda_vgg = cfg.SOLVER.LOSS.LAMBDA_VGG
        self.lambda_cos = cfg.SOLVER.LOSS.LAMBDA_COS
        self.lambda_spline = cfg.SOLVER.LOSS.LAMBDA_SPLINE
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

        enhance_img, spline = self.g_model(device_input)

        pixel_loss = self.pixel_loss(enhance_img, device_gt)
        ssim_loss = 1.0 - self.ssim_loss(enhance_img, device_gt)
        vgg_loss = self.vgg_loss(enhance_img, device_gt)
        cos_loss = 1 - torch.mean(self.cos_loss(enhance_img, device_gt))

        total_loss = self.lambda_pixel * pixel_loss + self.lambda_ssim * ssim_loss + self.lambda_vgg * vgg_loss + self.lambda_spline * spline + self.lambda_cos * cos_loss

        self.g_optimizer.zero_grad()
        total_loss.backward()
        self.g_optimizer.step()

        return {'total_loss': total_loss.item(),
                'pixel_loss': pixel_loss.item(),
                'ssim_loss': ssim_loss.item(),
                'vgg_loss': vgg_loss.item(),
                'cos_loss': cos_loss.item(),
                'spline': spline.item()}

    def generator(self, input_data):
        if self.pyramid is None:
            return self.g_model(input_data.to(self.device, non_blocking=True))

        input_data = input_data.to(self.device, non_blocking=True)
        input_lap, _ = self.pyramid.pyramid_decompose(input_data)
        low_res, cure_param = self.g_model(input_lap[-1])
        input_lap[-1] = low_res
        return self.pyramid.pyramid_recompose(input_lap), cure_param

