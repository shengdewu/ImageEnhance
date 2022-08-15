import torch.optim
from engine.model.gan_model import BaseGanModel
from engine.slover import build_lr_scheduler, build_optimizer_with_gradient_clipping
from engine.slover.lr_scheduler import EmptyLRScheduler
from typing import Iterator
import logging
from codes.discriminator.build import build_discriminator


class GanBaseModel(BaseGanModel):
    def __init__(self, cfg):
        super(GanBaseModel, self).__init__(cfg)
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create model {}'.format(self.__class__))
        self.device = cfg.MODEL.DEVICE
        return

    def create_d_optimizer(self, cfg, parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR if cfg.SOLVER.OPTIMIZER.get('D_LR', None) is None else cfg.SOLVER.OPTIMIZER.D_LR
        return build_optimizer_with_gradient_clipping(cfg, torch.optim.Adam)(
            parameters,
            lr=lr,
            betas=(cfg.SOLVER.OPTIMIZER.ADAM.B1, cfg.SOLVER.OPTIMIZER.ADAM.B2),
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
        )

    def create_g_optimizer(self, cfg, parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return build_optimizer_with_gradient_clipping(cfg, torch.optim.Adam)(
            parameters,
            lr=cfg.SOLVER.OPTIMIZER.BASE_LR,
            betas=(cfg.SOLVER.OPTIMIZER.ADAM.B1, cfg.SOLVER.OPTIMIZER.ADAM.B2),
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
        )

    def create_d_scheduler(self, cfg, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return EmptyLRScheduler(optimizer)

    def create_g_scheduler(self, cfg, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return build_lr_scheduler(cfg, optimizer)

    def create_g_model(self, cfg) -> torch.nn.Module:
        raise NotImplemented('the run_step must be implement')

    def create_d_model(self, cfg) -> torch.nn.Module:
        return build_discriminator(cfg)

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        """
        raise NotImplemented('the run_step must be implement')

    def generator(self, data):
        return self.g_model(data.to(self.device, non_blocking=True))


