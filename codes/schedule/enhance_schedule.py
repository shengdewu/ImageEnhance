from engine.schedule.scheduler import BaseScheduler
from codes.train.train import EnhanceTrainer


class EnhanceScheduler(BaseScheduler):
    def __init__(self):
        super(EnhanceScheduler, self).__init__()
        return

    def lunch_func(self, cfg, args):
        trainer = EnhanceTrainer(cfg)
        trainer.resume_or_load(args.resume)
        trainer.loop()
        return
