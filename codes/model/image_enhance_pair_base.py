import torch.optim
from engine.model.model import BaseModel
from engine.slover import build_lr_scheduler, build_optimizer_with_gradient_clipping
from engine.model.init_model import select_weights_init
from typing import Iterator
import logging
from codes.network.build import build_generator


class PairBaseModel(BaseModel):
    def __init__(self, cfg):
        super(PairBaseModel, self).__init__(cfg)
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create model {}'.format(self.__class__))

        self.device = cfg.MODEL.DEVICE
        return

    def create_optimizer(self, cfg, parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return build_optimizer_with_gradient_clipping(cfg, torch.optim.Adam)(
            self.model.parameters(),
            lr=cfg.SOLVER.OPTIMIZER.BASE_LR,
            betas=(cfg.SOLVER.OPTIMIZER.ADAM.B1, cfg.SOLVER.OPTIMIZER.ADAM.B2),
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
        )

    def create_scheduler(self, cfg, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return build_lr_scheduler(cfg, self.optimizer)

    def create_model(self, cfg) -> torch.nn.Module:
        model = build_generator(cfg)
        if cfg.MODEL.get('WEIGHTS_INIT_TYPE', 'none') != 'none':
            model.apply(select_weights_init(cfg.MODEL.WEIGHTS_INIT_TYPE)) #初始化和对应的激活函数有关系
        return model

    def run_step(self, data, *, epoch=None, **kwargs):
        raise NotImplemented('the run_step must be implement')

    def generator(self, input_data):
        return self.model(input_data.to(self.device, non_blocking=True))
