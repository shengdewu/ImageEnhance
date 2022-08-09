import torch
import engine.checkpoint.functional as checkpoint_f
from codes.model.build import BUILD_MODEL_REGISTRY
import math
from .image_enhance_pair_base import PairBaseModel
from codes.network.lut.loss.tv import TV3D
import logging
import copy
from engine.comm import TORCH_VERSION
from engine.loss.vgg_loss import PerceptualLoss
import torch.nn.functional as torch_func
from ..network.lut.trilinear.trilinear import TrilinearInterpolationModel


@BUILD_MODEL_REGISTRY.register()
class LutModel(PairBaseModel):
    def __init__(self, cfg):
        super(LutModel, self).__init__(cfg)

        self.lambda_class_smooth = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_CLASS_SMOOTH
        self.lambda_smooth = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_SMOOTH
        self.lambda_monotonicity = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_MONOTONICITY
        self.lambda_pixel = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_PIXEL
        self.lambda_perceptual = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_PERCEPTUAL

        self.criterion_perceptual = PerceptualLoss(cfg.MODEL.VGG.LAYER, device=self.device, path=cfg.MODEL.VGG.PATH)
        self.criterion_pixel_wise = torch.nn.MSELoss().to(self.device)
        self.tv3 = TV3D(cfg.MODEL.NETWORK.LUT.DIMS, cfg.MODEL.DEVICE)

        self.down_factor = cfg.INPUT.get('DOWN_FACTOR', 1)
        assert self.down_factor % 2 == 0 or self.down_factor == 1, 'the {} must be divisible by 2 or equal 1'.format(self.down_factor)

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
        if self.down_factor > 1:
            device_input = torch_func.interpolate(device_input, scale_factor=1 / self.down_factor, mode='bilinear')
            device_gt = torch_func.interpolate(device_gt, scale_factor=1 / self.down_factor, mode='bilinear')

        enhance_img, weights_norm = self.g_model(device_input)

        loss_pixel = self.criterion_pixel_wise(enhance_img, device_gt)
        loss_perceptual = self.criterion_perceptual(enhance_img, device_gt)

        tv_cons, mn_cons = self.g_model.calc_tv(self.tv3)

        total_loss = self.lambda_pixel * loss_pixel + self.lambda_smooth * tv_cons + self.lambda_class_smooth * weights_norm + self.lambda_monotonicity * mn_cons + self.lambda_perceptual * loss_perceptual

        psnr_avg = 10 * math.log10(1 / loss_pixel.item())

        self.g_optimizer.zero_grad()
        total_loss.backward()
        self.g_optimizer.step()

        return {'psnr_avg': psnr_avg,
                'mse_avg': loss_pixel.item(),
                'tv_cons': tv_cons.item(),
                'weights_norm': weights_norm.item(),
                'mn_cons': mn_cons.item(),
                'loss_perceptual': loss_perceptual.item()}

    def generator(self, input_data):
        x = input_data.to(self.device, non_blocking=True)
        dx = x
        if self.down_factor > 1:
            dx = torch_func.interpolate(dx, scale_factor=1 / self.down_factor, mode='bilinear')
        luts = self.g_model.lut(dx)

        fakes = list()
        for i in range(len(luts)):
            _, fake = TrilinearInterpolationModel()(luts[i], x[i, :].unsqueeze(0))
            fakes.append(fake)
        return torch.cat(fakes, dim=0)

    def enable_distribute(self, cfg):
        if cfg.MODEL.TRAINER.TYPE == 1 and cfg.MODEL.TRAINER.GPU_ID >= 0:
            logging.getLogger(self.default_log_name).info('launch model by distribute in gpu_id {}'.format(cfg.MODEL.TRAINER.GPU_ID))
            self.g_model.enable_model_distributed()
        elif cfg.MODEL.TRAINER.TYPE == 0:
            logging.getLogger(self.default_log_name).info('launch model by parallel')
            self.g_model = torch.nn.parallel.DataParallel(self.g_model)
        else:
            logging.getLogger(self.default_log_name).info('launch model by stand alone machine')

    def load_state_dict(self, state_dict: dict):
        def _match_model_state_dict(model, checkpoint, log_name=None):
            checkpoint_state_dict = copy.deepcopy(checkpoint)
            model_state_dict = checkpoint_f.get_model_state_dict(model)

            incorrect_shapes = []
            missing_keys = []
            for sk, state_dicts in checkpoint.items():
                model_state_dict_at_sk = model_state_dict.get(sk, None)
                if model_state_dict_at_sk is None:
                    checkpoint_state_dict.pop(sk)
                    missing_keys.append(sk)
                    continue
                for k, state in state_dicts.items():
                    model_state = model_state_dict_at_sk.get(k, None)
                    if model_state is None:
                        checkpoint_state_dict[sk].pop(k)
                        missing_keys.append('{}/{}'.format(sk, k))
                        continue
                    for name in checkpoint_state_dict[sk][k].keys():
                        if name in model_state.keys():
                            model_param = model_state[name]
                            # Allow mismatch for uninitialized parameters
                            if TORCH_VERSION >= (1, 8) and isinstance(
                                    model_param, torch.nn.parameter.UninitializedParameter
                            ):
                                continue
                            shape_model = tuple(model_param.shape)
                            shape_checkpoint = tuple(checkpoint_state_dict[sk][k][name].shape)
                            if shape_model != shape_checkpoint:
                                incorrect_shapes.append(('{}/{}/{}'.format(sk, k, name), shape_checkpoint, shape_model))
                                checkpoint_state_dict[sk][k].pop(name)
                        else:
                            checkpoint_state_dict[sk][k].pop(name)
                            missing_keys.append('{}/{}/{}'.format(sk, k, name))

            if log_name is not None:
                if missing_keys:
                    logging.getLogger(log_name).warning('the model missing_keys keys \n {}'.format(missing_keys))
                if incorrect_shapes:
                    logging.getLogger(log_name).warning('the model incorrect_shapes keys \n {}'.format(incorrect_shapes))

            return checkpoint_state_dict

        match_state_dict = _match_model_state_dict(self.g_model, state_dict['g_model'], log_name=self.default_log_name)
        if isinstance(self.g_model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            self.g_model.module.load_state_dict(match_state_dict)
        else:
            self.g_model.load_state_dict(match_state_dict)
        return
