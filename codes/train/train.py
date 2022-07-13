from engine.trainer.trainer import BaseTrainer
from engine.model.model import BaseModel
from engine.log.logger import setup_logger
import engine.comm as comm
from codes.model.build import build_model
from codes.data.build import build_dataset
import codes.data.fn.collate_fn as collate_fn
import engine.data.data_loader as engine_build_loader

import logging
import torch
import codes.train.train_fn as train_fn


class EnhanceTrainer(BaseTrainer):
    criterion_pixel_wise = torch.nn.MSELoss()

    def __init__(self, cfg):
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)
        super(EnhanceTrainer, self).__init__(cfg)

        train_dataset = build_dataset(cfg, 'train')
        test_dataset = build_dataset(cfg, 'test')

        logging.getLogger(__name__).info('create dataset {}  then load {} train data, load {} test data'.format(cfg.DATALOADER.DATASET, len(train_dataset), len(test_dataset)))

        if cfg.MODEL.TRAINER.TYPE == 1 and cfg.MODEL.TRAINER.GPU_ID >= 0:
            self.data_loader = engine_build_loader.create_distribute_iterable_data_loader(dataset=train_dataset,
                                                                                          batch_size=cfg.SOLVER.TRAIN_PER_BATCH,
                                                                                          rank=cfg.MODEL.TRAINER.GLOBAL_RANK,
                                                                                          world_size=cfg.MODEL.TRAINER.WORLD_SIZE,
                                                                                          num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                                                          collate_fn=collate_fn.fromlist)
        else:
            self.data_loader = engine_build_loader.create_iterable_data_loader(dataset=train_dataset,
                                                                               batch_size=cfg.SOLVER.TRAIN_PER_BATCH,
                                                                               num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                                               collate_fn=collate_fn.fromlist)

        self.test_data_loader = engine_build_loader.create_data_loader(dataset=test_dataset, batch_size=cfg.SOLVER.TEST_PER_BATCH, num_workers=1, collate_fn=collate_fn.fromlist)

        self.total_data_per_epoch = len(train_dataset) / cfg.SOLVER.TRAIN_PER_BATCH
        self.iter_train_loader = iter(self.data_loader)
        logging.getLogger(__name__).info('ready for training : there are {} data in one epoch and actually trained for {} epoch'.format(self.total_data_per_epoch, self.max_iter / self.total_data_per_epoch))
        return

    def create_model(self, cfg) -> BaseModel:
        return build_model(cfg)

    def loop(self):
        psnr = train_fn.calculate_psnr(self.model, self.test_data_loader, self.device, EnhanceTrainer.criterion_pixel_wise)
        logging.getLogger(__name__).info('before train psnr = {}'.format(psnr))

        self.model.enable_train()

        for epoch in range(self.start_iter, self.max_iter):
            data = next(self.iter_train_loader)

            loss_dict = self.model(data, epoch=epoch)

            self.checkpoint.save(self.model, epoch)
            self.run_after(epoch, loss_dict)

        if self.start_iter < self.max_iter:
            self.checkpoint.save(self.model, self.max_iter)

        psnr = train_fn.calculate_psnr(self.model, self.test_data_loader, self.device, EnhanceTrainer.criterion_pixel_wise)
        logging.getLogger(__name__).info('after train psnr = {}'.format(psnr))
        train_fn.visualize_result(self.model, self.test_data_loader, self.device, self.output, EnhanceTrainer.criterion_pixel_wise)
        return

    def run_after(self, epoch, loss_dict):
        if int(epoch+0.5) % self.checkpoint.check_period == 0:
            logging.getLogger(__name__).info('trainer run step {} : {}'.format(epoch, loss_dict))
        return


