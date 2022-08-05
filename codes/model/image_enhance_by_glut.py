import torch
import engine.checkpoint.functional as checkpoint_f
from codes.model.build import BUILD_MODEL_REGISTRY
import math
from .image_enhance_gan_base import GanBaseModel
from codes.network.lut.loss.tv import TV3D
import logging
import copy
from codes.losses.gradient_penalty import compute_gradient_penalty
from engine.comm import TORCH_VERSION
from codes.network.build import build_generator
from engine.model.init_model import select_weights_init


@BUILD_MODEL_REGISTRY.register()
class LutGanModel(GanBaseModel):
    def __init__(self, cfg):
        super(LutGanModel, self).__init__(cfg)

        self.lambda_class_smooth = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_CLASS_SMOOTH
        self.lambda_smooth = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_SMOOTH
        self.lambda_monotonicity = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_MONOTONICITY
        self.lambda_pixel = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_PIXEL

        self.lambda_gp = cfg.SOLVER.LOSS.LAMBDA.LAMBDA_GP
        self.n_critic = cfg.SOLVER.LOSS.LAMBDA.N_CRITIC

        self.criterion_pixel_wise = torch.nn.MSELoss().to(self.device)
        self.tv3 = TV3D(cfg.MODEL.NETWORK.LUT.DIMS, cfg.MODEL.DEVICE)

        return

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        """
        real_a = data['input'].to(self.device, non_blocking=True)
        real_b = data['expert'].to(self.device, non_blocking=True)

        self.d_optimizer.step()

        fake_b, weights_norm = self.g_model(real_a)

        loss_dict = dict()
        real_pred = self.d_model(real_b)
        fake_pred = self.d_model(fake_b)

        gradient_penalty = compute_gradient_penalty(self.d_model, real_b, fake_b, grad_outputs_shape=real_pred.shape, device=self.device)

        loss_d = -torch.mean(real_pred) + torch.mean(fake_pred) + self.lambda_gp * gradient_penalty

        loss_d.backward()
        self.d_optimizer.step()

        loss_dict['d_loss'] = loss_d.item()

        if epoch % self.n_critic == 0:
            self.g_optimizer.step()

            fake_b, weights_norm = self.g_model(real_a)
            fake_pred = self.d_model(fake_b)
            loss_pixel = self.criterion_pixel_wise(fake_b, real_a)
            tv_cons, mn_cons = self.g_model.calc_tv(self.tv3)

            loss_g = -torch.mean(fake_pred) + self.lambda_pixel * loss_pixel + self.lambda_smooth * (weights_norm + tv_cons) + self.lambda_monotonicity * mn_cons

            loss_g.backward()
            self.g_optimizer.step()

            loss_dict['loss_g'] = loss_g.item()
            loss_dict['loss_pixel'] = loss_pixel.item()
            loss_dict['weights_norm'] = weights_norm.item()
            loss_dict['tv_cons'] = tv_cons.item()
            loss_dict['mn_cons'] = mn_cons.item()
            loss_dict['psnr_avg'] = 10 * math.log10(1 / loss_pixel.item())

        return loss_dict

    def create_g_model(self, cfg) -> torch.nn.Module:
        model = build_generator(cfg)
        if cfg.MODEL.get('WEIGHTS_INIT_TYPE', 'none') != 'none':
            model.apply(select_weights_init(cfg.MODEL.WEIGHTS_INIT_TYPE)) #初始化和对应的激活函数有关系
        return model

    def generator(self, input_data):
        return self.g_model(input_data.to(self.device, non_blocking=True))

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
