import torch
from codes.model.build import BUILD_MODEL_REGISTRY
import math
from .image_enhance_pair_base import PairBaseModel
from codes.network.lut.loss.tv import TV3D


@BUILD_MODEL_REGISTRY.register()
class LutModel(PairBaseModel):
    def __init__(self, cfg):
        super(LutModel, self).__init__(cfg)

        self.lambda_class_smooth = cfg.SOLVER.LOSS.LAMBDA_CLASS_SMOOTH
        self.lambda_smooth = cfg.SOLVER.LOSS.LAMBDA_SMOOTH
        self.lambda_monotonicity = cfg.SOLVER.LOSS.LAMBDA_MONOTONICITY
        self.lambda_pixel = cfg.SOLVER.LOSS.LAMBDA_PIXEL

        self.criterion_pixel_wise = torch.nn.MSELoss().to(self.device)
        self.tv3 = TV3D(cfg.MODEL.LUT.DIMS, cfg.MODEL.DEVICE)

        return

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        """
        device_input = data['input'].to(self.device, non_blocking=True)

        enhance_img, weights_norm = self.g_model(device_input)

        device_gt = data['expert'].to(self.device, non_blocking=True)

        loss_pixel = self.criterion_pixelwise(enhance_img, device_gt)

        tv_cons, mn_cons = self.g_model(self.tv3)

        total_loss = self.lambda_pixel * loss_pixel + self.lambda_smooth * (weights_norm + tv_cons) + self.lambda_monotonicity * mn_cons

        psnr_avg = 10 * math.log10(1 / loss_pixel.item())

        self.optimizer_G.step()

        self.g_optimizer.zero_grad()
        total_loss.backward()
        self.g_optimizer.step()

        return {'psnr_avg': psnr_avg,
                'mse_avg': loss_pixel.item(),
                'tv_cons': tv_cons.item(),
                'weights_norm': weights_norm.item(),
                'mn_cons': mn_cons.item()}

    def generator(self, input_data):
        return self.g_model(input_data.to(self.device, non_blocking=True))

