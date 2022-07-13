import torch
from codes.model.build import BUILD_MODEL_REGISTRY
from engine.loss.vgg_loss import PerceptualLoss
import engine.loss.ssim_loss as engine_ssim
from .image_enhance_pair_base import PairBaseModel
from codes.data.fn import rgb2luma
from codes.network.curl.curl_luma import CurlLumaNet


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
        self.model_is_luma = self.model.__class__.__name__ == CurlLumaNet.__name__
        return

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        """
        device_input = data['input'].to(self.device, non_blocking=True)

        if self.model_is_luma:
            device_gray = rgb2luma.rgb2luma_bt601_nchw(device_input).unsqueeze(1)
            # device_gray = data['luma'].to(self.device, non_blocking=True)
            enhance_img, spline = self.model(device_input, device_gray)
        else:
            enhance_img, spline = self.model(device_input)

        device_gt = data['expert'].to(self.device, non_blocking=True)

        pixel_loss = self.pixel_loss(enhance_img, device_gt)
        ssim_loss = 1.0 - self.ssim_loss(enhance_img, device_gt)
        vgg_loss = self.vgg_loss(enhance_img, device_gt)
        cos_loss = 1 - torch.mean(self.cos_loss(enhance_img, device_gt))

        total_loss = self.lambda_pixel * pixel_loss + self.lambda_ssim * ssim_loss + self.lambda_vgg * vgg_loss + self.lambda_spline * spline + self.lambda_cos * cos_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {'total_loss': total_loss.item(),
                'pixel_loss': pixel_loss.item(),
                'ssim_loss': ssim_loss.item(),
                'vgg_loss': vgg_loss.item(),
                'cos_loss': cos_loss.item(),
                'spline': spline.item()}

    def generator(self, input_data):
        if self.model_is_luma:
            gray_data = rgb2luma.rgb2luma_bt601_nchw(input_data).unsqueeze(1)
            return self.model(input_data.to(self.device, non_blocking=True), gray_data.to(self.device, non_blocking=True))
        else:
            return self.model(input_data.to(self.device, non_blocking=True))

