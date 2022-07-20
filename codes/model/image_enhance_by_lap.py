import torch
from codes.model.image_enhance_gan_base import GanBaseModel
from engine.model.init_model import select_weights_init
from codes.losses.losses import MSELoss, GANLoss
from codes.model.build import BUILD_MODEL_REGISTRY
from codes.network.build import build_generator
from codes.losses.gradient_penalty import compute_gradient_penalty
from codes.network.lap_pyr.lap_pyramid import LapPyramidConv


@BUILD_MODEL_REGISTRY.register()
class LapGanModel(GanBaseModel):
    def __init__(self, cfg):
        super(LapGanModel, self).__init__(cfg)

        self.pixel_loss = MSELoss(loss_weight=cfg.SOLVER.LOSS.LAMBDA.LAMBDA_PIXEL)
        self.gan_loss = GANLoss(gan_type=cfg.SOLVER.LOSS.get('GAN_TYPE', 'lsgan'), loss_weight=cfg.SOLVER.LOSS.LAMBDA.LAMBDA_GAN)
        self.lambda_gp = cfg.SOLVER.LOSS.LAMBDA.get('LAMBDA_GP', 100)

        self.net_d_iters = cfg.SOLVER.LOSS.get('DISCRIMINATOR_ITERS', 1)
        self.net_d_init_iters = cfg.SOLVER.LOSS.get('DISCRIMINATOR_INIT_ITERS', 0)

        level = 3
        if cfg.MODEL.NETWORK.get('LAP_PYRAMID', None) is not None:
            level = cfg.MODEL.NETWORK.LAP_PYRAMID.get('PYRAMID_LEVEL', level)
        self.pyramid = LapPyramidConv(level)
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

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        """
        input_img = data['input'].to(self.device, non_blocking=True)
        ref_img = data['ref'].to(self.device, non_blocking=True)

        input_lap, _ = self.pyramid_decom(input_img)
        ref_lap, ref_gauss = self.pyramid_decom(ref_img)
        input_recons = self.pyramid_recons(input_lap)
        input_lap[-1] = ref_lap[-1]
        input_ref_recons = self.pyramid_recons(input_lap)
        import cv2, torchvision
        import numpy as np

        def save(tensor, fp):
            fmt = np.uint8
            grid = torchvision.utils.make_grid(tensor)
            # Add 0.5 after unnormalizing to [0, unnormalizing_value] to round to nearest integer
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu').numpy().astype(fmt)
            cv2.imwrite(fp, ndarr[:, :, ::-1])
        save(torch.cat((input_img, input_recons, input_ref_recons, ref_img), dim=0), 'pyr{}.jpg'.format(int(np.random.rand()*100)))

        ref_img = ref_gauss[-1].clone()
        del ref_gauss

        loss_dict = dict()
        # optimize g
        for p in self.d_model.parameters():
            p.requires_grad = False

        self.g_optimizer.zero_grad()
        fake = self.g_model(input_lap[-1])

        l_g_total = torch.zeros(1).to(self.device)
        if epoch % self.net_d_iters == 0 and epoch > self.net_d_init_iters:
            l_g_pix = self.pixel_loss(fake, input_img)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix.item()

            fake_g = self.d_model(fake)
            l_g_gan = self.gan_loss(fake_g, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan.item()

            l_g_total.backward()
            self.g_optimizer.step()

        # optimize d
        for p in self.d_model.parameters():
            p.requires_grad = True

        self.d_optimizer.zero_grad()
        fake = self.g_model(input_img)
        # real
        d_real = self.d_model(ref_img)
        l_d_real = self.gan_loss(d_real, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real.item()
        # fake
        d_fake = self.d_model(fake)
        l_d_fake = self.gan_loss(d_fake, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake.item()

        gp = compute_gradient_penalty(self.d_model, ref_img, fake, d_real.shape)
        l_d_loss = l_d_real + l_d_fake + self.lambda_gp * gp
        l_d_loss.backward()
        self.d_optimizer.step()

        return loss_dict

    def generator(self, data):
        input_data = data.to(self.device, non_blocking=True)
        input_lap, _ = self.pyramid_decom(input_data)
        low_res = self.g_model(input_lap[-1])
        input_lap[-1] = low_res
        return self.pyramid_recons[input_lap]

