import torch
from codes.model.image_enhance_gan_base import GanBaseModel
from engine.model.init_model import select_weights_init
from codes.losses.losses import MSELoss, GANLoss
from codes.model.build import BUILD_MODEL_REGISTRY
from codes.network.build import build_generator
from codes.losses.gradient_penalty import compute_gradient_penalty
from codes.network.lap_pyr.lap_pyramid import LapPyramidConv
import logging


@BUILD_MODEL_REGISTRY.register()
class LapGanModel(GanBaseModel):
    def __init__(self, cfg):
        super(LapGanModel, self).__init__(cfg)

        self.pixel_loss = MSELoss(loss_weight=cfg.SOLVER.LOSS.LAMBDA.LAMBDA_PIXEL)
        self.gan_loss = GANLoss(gan_type=cfg.SOLVER.LOSS.get('GAN_TYPE', 'lsgan'), loss_weight=cfg.SOLVER.LOSS.LAMBDA.LAMBDA_GAN)
        self.lambda_gp = cfg.SOLVER.LOSS.LAMBDA.get('LAMBDA_GP', 100)

        self.net_d_iters = cfg.SOLVER.LOSS.get('DISCRIMINATOR_ITERS', 1)
        self.net_d_init_iters = cfg.SOLVER.LOSS.get('DISCRIMINATOR_INIT_ITERS', 0)

        self.level = 3
        self.train_level = 1
        if cfg.MODEL.NETWORK.get('LAP_PYRAMID', None) is not None:
            self.level = cfg.MODEL.NETWORK.LAP_PYRAMID.get('PYRAMID_LEVEL', self.level)
            self.train_level = cfg.MODEL.NETWORK.LAP_PYRAMID.get('TRAIN_LEVEL', self.train_level)
        assert self.train_level <= self.level, 'the train_level must be smaller level, but train_level {}, level {}'.format(self.train_level, self.level)

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('param: \nGAN_TYPE: {} \nLAMBDA_GP: {} \nDISCRIMINATOR_ITERS: {}\nDISCRIMINATOR_INIT_ITERS:{} \nPYRAMID_LEVEL: {}\nTRAIN_LEVEL: {}'.format(cfg.SOLVER.LOSS.get('GAN_TYPE', 'lsgan'),
                                                                                                                                                                                               cfg.SOLVER.LOSS.LAMBDA.get('LAMBDA_GP', 100),
                                                                                                                                                                                               self.net_d_iters,
                                                                                                                                                                                               self.net_d_init_iters,
                                                                                                                                                                                               self.level,
                                                                                                                                                                                               self.train_level))

        self.pyramid = LapPyramidConv(self.level)
        return

    def create_g_model(self, cfg) -> torch.nn.Module:
        model = build_generator(cfg)
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
        g_loss = self.pixel_loss(fakes[0], ref_gauss[-1])
        for i in range(1, self.train_level):
            g_loss += self.pixel_loss(fakes[i], ref_gauss[-(i+1)])
        return g_loss

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

        loss_dict = dict()
        # optimize g
        for p in self.d_model.parameters():
            p.requires_grad = False

        self.g_optimizer.zero_grad()

        fakes = self.pyramid_generator(input_lap)

        l_g_total = torch.zeros(1).to(self.device)
        if epoch % self.net_d_iters == 0 and epoch > self.net_d_init_iters:
            l_g_pix = self.pyramid_g_loss(fakes, ref_gauss)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix.item()

            fake_g = self.d_model(fakes[-1])
            l_g_gan = self.gan_loss(fake_g, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan.item()

            l_g_total.backward()
            self.g_optimizer.step()

        # optimize d
        for p in self.d_model.parameters():
            p.requires_grad = True

        self.d_optimizer.zero_grad()
        fakes = self.pyramid_generator(input_lap)
        # real
        d_real = self.d_model(ref_gauss[-self.train_level])
        l_d_real = self.gan_loss(d_real, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real.item()
        # fake
        d_fake = self.d_model(fakes[-1])
        l_d_fake = self.gan_loss(d_fake, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake.item()

        gp = compute_gradient_penalty(self.d_model, ref_gauss[-self.train_level], fakes[-1], d_real.shape)
        l_d_loss = l_d_real + l_d_fake + self.lambda_gp * gp
        l_d_loss.backward()
        self.d_optimizer.step()

        return loss_dict

    def generator(self, data):
        input_data = data.to(self.device, non_blocking=True)
        input_lap, _ = self.pyramid_decom(input_data)
        low_res = self.g_model(input_lap[-1])
        input_lap[-1] = low_res
        return self.pyramid_recons(input_lap)

