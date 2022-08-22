import torch
from codes.model.image_enhance_pair_base import PairBaseModel
from engine.model.init_model import select_weights_init
from engine.loss.vgg_loss import PerceptualLoss
import engine.loss.ssim_loss as engine_ssim
from codes.model.build import BUILD_MODEL_REGISTRY
from codes.network.lap_pyr.lap_pyramid import LapPyramidConv
import logging
from codes.network.maxim.config import maxim_config
from codes.network.maxim.maxim import MAXIM


@BUILD_MODEL_REGISTRY.register()
class MaximModel(PairBaseModel):
    def __init__(self, cfg):
        super(MaximModel, self).__init__(cfg)

        self.vgg_loss = PerceptualLoss(cfg.MODEL.VGG.LAYER, device=self.device, path=cfg.MODEL.VGG.PATH)
        self.ssim_loss = engine_ssim.SSIM()
        self.pixel_loss = torch.nn.L1Loss().to(self.device)

        self.lambda_ssim = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_SSIM
        self.lambda_vgg = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_VGG
        self.lambda_pixel = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_PIXEL

        self.level = 3
        self.train_level = 1
        if cfg.MODEL.NETWORK.get('LAP_PYRAMID', None) is not None:
            self.level = cfg.MODEL.NETWORK.LAP_PYRAMID.get('PYRAMID_LEVEL', self.level)
            self.train_level = cfg.MODEL.NETWORK.LAP_PYRAMID.get('TRAIN_LEVEL', self.train_level)

        assert self.train_level <= self.level, 'the train_level must be smaller level, but train_level {}, level {}'.format(self.train_level, self.level)

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('param: \nVGG.LAYER: {} \nVGG.PATH: {} '
                                                    '\nLAMBDA_SSIM:{} \nLAMBDA_VGG: {}'
                                                    '\nLAMBDA_PIXEL: {}\nTRAIN_LEVEL: {}'
                                                    '\nPYRAMID_LEVEL {}'.format(cfg.MODEL.VGG.LAYER,
                                                                                cfg.MODEL.VGG.PATH,
                                                                                self.lambda_ssim,
                                                                                self.lambda_vgg,
                                                                                self.lambda_pixel,
                                                                                self.train_level,
                                                                                self.level))

        self.pyramid = LapPyramidConv(self.level)
        return

    def create_g_model(self, cfg) -> torch.nn.Module:
        config = maxim_config[cfg.MODEL.NETWORK.ARCH]
        model = MAXIM(**config)
        if cfg.MODEL.get('WEIGHTS_INIT_TYPE', 'none') != 'none':
            model.apply(select_weights_init(cfg.MODEL.WEIGHTS_INIT_TYPE)) #初始化和对应的激活函数有关系
        return model

    @torch.no_grad()
    def pyramid_decom(self, img):
        return self.pyramid.pyramid_decom(img)

    @torch.no_grad()
    def pyramid_recons(self, pyramid):
        return self.pyramid.pyramid_recons(pyramid)

    def pyramid_generator(self, lap_pyramid):
        fakes = [self.g_model(lap_pyramid[-1])]
        for i in range(1, self.train_level):
            pyramid = [lap_pyramid[-(i+1)], fakes[i-1]]
            i_lap = self.pyramid.pyramid_recons(pyramid)
            fakes.append(self.g_model(i_lap))
        return fakes

    def pyramid_g_loss(self, fakes, ref_gauss):
        pixel_loss = self.pixel_loss(fakes[0], ref_gauss[-1])
        ssim_loss = 1.0 - self.ssim_loss(fakes[0], ref_gauss[-1])
        vgg_loss = self.vgg_loss(fakes[0], ref_gauss[-1])
        for i in range(1, self.train_level):
            pixel_loss += self.pixel_loss(fakes[i], ref_gauss[-(i+1)])
            ssim_loss += 1.0 - self.ssim_loss(fakes[i], ref_gauss[-(i+1)])
            vgg_loss += self.vgg_loss(fakes[i], ref_gauss[-(i+1)])

        return pixel_loss, ssim_loss, vgg_loss

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        """
        input_img = data['input'].to(self.device, non_blocking=True)
        ref_img = data['expert'].to(self.device, non_blocking=True)

        input_lap, _ = self.pyramid_decom(input_img)
        _, ref_gauss = self.pyramid_decom(ref_img)

        fakes = self.pyramid_generator(input_lap)

        pixel_loss, ssim_loss, vgg_loss = self.pyramid_g_loss(fakes, ref_gauss)

        total_loss = self.lambda_pixel * pixel_loss + self.lambda_ssim * ssim_loss + self.lambda_vgg * vgg_loss

        self.g_optimizer.zero_grad()
        total_loss.backward()
        self.g_optimizer.step()

        return {'total_loss': total_loss.item(),
                'pixel_loss': pixel_loss.item(),
                'ssim_loss': ssim_loss.item(),
                'vgg_loss': vgg_loss.item()
                }

    def generator(self, data):
        input_data = data.to(self.device, non_blocking=True)
        input_lap, _ = self.pyramid_decom(input_data)
        low_res = self.g_model(input_lap[-1])
        input_lap[-1] = low_res
        return self.pyramid_recons(input_lap)

