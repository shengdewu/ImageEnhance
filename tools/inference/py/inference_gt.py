import torch
import os
import engine.comm as comm
from engine.log.logger import setup_logger
import tqdm
from .inference import Inference
from codes.data.build import build_dataset
import logging
from codes.train.train_fn import save_image


class InferenceNoneGt(Inference):
    def __init__(self, cfg, tif=True):
        super(InferenceNoneGt, self).__init__(cfg, tif)
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)
        return

    def loop(self, cfg, skip=False, special_name=None):
        if special_name is not None:
            assert (isinstance(special_name, list) or isinstance(special_name, tuple)) and len(special_name) > 0

        output = cfg.OUTPUT_DIR
        os.makedirs(output, exist_ok=True)

        test_dataset = build_dataset(cfg, mode='test')
        logging.getLogger(__name__).info('from {} create dataset {}, load {} test data'.format(cfg.DATALOADER.DATA_PATH, cfg.DATALOADER.DATASET, len(test_dataset)))

        img_format = 'jpg' if self.unnormalizing_value == 255 else 'tif'
        skin_name = list()
        if skip:
            skin_name = [name for name in os.listdir(output) if name.lower().endswith(img_format)]

        self.model.disable_train()
        for index in tqdm.tqdm(range(len(test_dataset))):
            data = test_dataset.get_item(index, skin_name, special_name=special_name, img_format=img_format)
            if data['input'] is None:
                continue

            real = data['input'].to(self.device).unsqueeze(0)
            with torch.no_grad():
                enhance_img = self.model.generator(real)
            if isinstance(enhance_img, list) or isinstance(enhance_img, tuple):
                enhance_img = enhance_img[0]

            img_sample = torch.cat((real, enhance_img), -1)

            mean_std = {'mean': cfg.INPUT.get('DATA_MEAN', None),
                        'std': cfg.INPUT.get('DATA_STD', None)}

            save_image(img_sample, '{}/{}'.format(output, data['name']), unnormalizing_value=self.unnormalizing_value, nrow=1, normalize=False, **mean_std)
        return


