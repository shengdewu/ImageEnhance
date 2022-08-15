import torch
from codes.model.image_enhance_gan_base import GanBaseModel
from engine.model.init_model import select_weights_init
from codes.losses.losses import MSELoss, GANLoss
from codes.model.build import BUILD_MODEL_REGISTRY
from codes.network.build import build_generator
from codes.losses.gradient_penalty import compute_gradient_penalty
from codes.network.lap_pyr.lap_pyramid import LapPyramidConv
import logging
import torch.nn.functional as torch_func


class DLoss(torch.nn.Module):
    def __init__(self):
        super(DLoss, self).__init__()
        return

    def forward(self, p_y, p_t):
        loss = -torch.mean(torch.log(torch.sigmoid(p_t) + 1e-9)) - torch.mean(torch.log(1 - torch.sigmoid(p_y) + 1e-9))
        return loss


class PyrLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(PyrLoss, self).__init__()
        self.weight = weight
        self.criterion = torch.nn.L1Loss(reduce='sum')
        return

    def forward(self, y_list, t_list):
        n = len(y_list)
        loss = torch.zeros(1, device=y_list[0].device)
        for m in range(n - 1):
            loss += self.weight * (2 ** (n - m - 2)) * self.criterion(
                y_list[m],
                torch_func.interpolate(t_list[m], (y_list[m].shape[2], y_list[m].shape[3]), mode='bilinear', align_corners=True)
            ) / y_list[m].shape[0]
        return loss


class RecLoss(torch.nn.Module):
    def __init__(self, weight=1.):
        super(RecLoss, self).__init__()
        self.weight = weight
        self.criterion = torch.nn.L1Loss(reduction='sum')

    def forward(self, y_list, t_list):
        loss = self.weight * self.criterion(y_list[-1], t_list[-1]) / y_list[-1].shape[0]
        return loss


class AdvLoss(torch.nn.Module):
    def __init__(self, size=256, weight=1.0):
        super(AdvLoss, self).__init__()
        self.weight = weight
        self.size = size
        return

    def forward(self, p_y):
        loss = -self.weight * 12 * self.size * self.size * torch.mean(torch.log(torch.sigmoid(p_y) + 1e-9))
        return loss


class Genloss(torch.nn.Module):
    def __init__(self, size=256, pyr_weight=1.0, rec_weight=1.0, adv_weight=1.0):
        super(Genloss, self).__init__()
        self.pyr_loss = PyrLoss(pyr_weight)
        self.rec_loss = RecLoss(rec_weight)
        self.adv_loss = AdvLoss(size, adv_weight)

    def forward(self, y_list, t_list, p_y=None, withoutadvloss=False):
        pyr_loss = self.pyr_loss(y_list, t_list)
        rec_loss = self.rec_loss(y_list, t_list)
        if withoutadvloss:
            loss = pyr_loss + rec_loss
            return rec_loss, pyr_loss, loss
        else:
            adv_loss = self.adv_loss(p_y)
            loss = pyr_loss + rec_loss + adv_loss
            return rec_loss, pyr_loss, adv_loss, loss


@BUILD_MODEL_REGISTRY.register()
class MENGanModel(GanBaseModel):
    def __init__(self, cfg):
        super(MENGanModel, self).__init__(cfg)
        self.use_level = 4

        self.dis_loss = DLoss().to(self.device)
        self.gen_loss = Genloss(size=256).to(self.device)

        self.lambda_gp = cfg.SOLVER.LOSS.LAMBDA.get('LAMBDA_GP', 100)
        self.enable_d_model = cfg.MODEL.get('ENABLE_DISCRIMINATOR', True)
        self.net_d_iters = cfg.SOLVER.LOSS.get('DISCRIMINATOR_ITERS', 1)
        self.net_d_init_iters = cfg.SOLVER.LOSS.get('DISCRIMINATOR_INIT_ITERS', 15)

        self.level = self.use_level + 1
        if cfg.MODEL.NETWORK.get('LAP_PYRAMID', None) is not None:
            self.level = cfg.MODEL.NETWORK.LAP_PYRAMID.get('PYRAMID_LEVEL', self.level)

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('param: \nGAN_TYPE:{}\nLAMBDA_GP:{}\nDISCRIMINATOR_ITERS: {}'
                                                    '\nDISCRIMINATOR_INIT_ITERS:{}\nENABLE_DISCRIMINATOR:{}'
                                                    '\nLAP_PYRAMID:{}'.format(cfg.SOLVER.LOSS.get('GAN_TYPE', 'lsgan'),
                                                                              cfg.SOLVER.LOSS.LAMBDA.get('LAMBDA_GP', 100),
                                                                              self.net_d_iters,
                                                                              self.net_d_init_iters,
                                                                              self.enable_d_model,
                                                                              self.level))

        self.pyramid = LapPyramidConv(self.level)
        return

    def create_g_model(self, cfg) -> torch.nn.Module:
        model = build_generator(cfg)
        if cfg.MODEL.get('WEIGHTS_INIT_TYPE', 'none') != 'none':
            model.apply(select_weights_init(cfg.MODEL.WEIGHTS_INIT_TYPE))  # 初始化和对应的激活函数有关系
        return model

    @torch.no_grad()
    def pyramid_de_com(self, img):
        return self.pyramid.pyramid_decom(img)

    @torch.no_grad()
    def pyramid_re_cons(self, pyramid):
        return self.pyramid.pyramid_recons(pyramid)

    def pyramid_g_loss(self, fakes, ref_gauss):
        g_loss = self.pixel_loss(fakes[0], ref_gauss[-1])
        for i in range(1, self.train_level):
            g_loss += self.pixel_loss(fakes[i], ref_gauss[-(i + 1)])
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

        input_lap, _ = self.pyramid_de_com(input_img)
        _, ref_gauss = self.pyramid_de_com(ref_img)

        # no_train_lap = input_lap[0: len(input_lap) - self.use_level]
        # no_train_ref = ref_gauss[0: len(input_lap) - self.use_level]

        input_lap = input_lap[len(input_lap) - self.use_level:]
        ref_gauss = ref_gauss[len(ref_gauss) - self.use_level:]
        ref_gauss.reverse()

        loss_dict = dict()
        # optimize d
        if self.enable_d_model:
            # for p in self.d_model.parameters():
            #     p.requires_grad = True

            self.d_optimizer.zero_grad()
            y_list = self.g_model(input_lap)
            y_list = [y.detach() for y in y_list]
            p_y = self.d_model(y_list[-1])
            p_t = self.d_model(ref_gauss[-1])
            d_loss = self.dis_loss(p_y, p_t)
            d_loss.backward()
            self.d_optimizer.step()
            loss_dict['d_loss'] = d_loss.item()

        # optimize g
        # for p in self.d_model.parameters():
        #     p.requires_grad = False

        self.g_optimizer.zero_grad()
        y_list = self.g_model(input_lap)
        if epoch > self.net_d_init_iters:
            if self.enable_d_model:
                p_y = self.d_model(y_list[-1])
                rec_loss, pyr_loss, adv_loss, loss = self.gen_loss(y_list, ref_gauss, p_y, withoutadvloss=False)
                loss_dict = {'rec_loss': rec_loss.item(), 'pyr_loss': pyr_loss.item(), 'adv_loss': adv_loss.item(), 'loss': loss.item()}
            else:
                rec_loss, pyr_loss, loss = self.gen_loss(y_list, ref_gauss, withoutadvloss=True)
                loss_dict = {'rec_loss': rec_loss.item(), 'pyr_loss': pyr_loss.item(), 'loss': loss.item()}
        else:
            rec_loss, pyr_loss, loss = self.gen_loss(y_list, ref_gauss, withoutadvloss=True)
            loss_dict = {'rec_loss': rec_loss.item(), 'pyr_loss': pyr_loss.item(), 'loss': loss.item()}

            loss.backward()
            self.g_optimizer.step()

        return loss_dict

    def generator(self, data):
        input_data = data.to(self.device, non_blocking=True)
        input_lap, _ = self.pyramid_de_com(input_data)
        no_train_lap = input_lap[0: len(input_lap) - self.use_level]
        input_lap = input_lap[len(input_lap) - self.use_level:]
        y_list = self.g_model(input_lap)
        no_train_lap.append(y_list[-1])
        return self.pyramid_re_cons(no_train_lap)
