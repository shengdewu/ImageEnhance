import logging
from engine.checkpoint.checkpoint_state_dict import CheckPointStateDict
from codes.model.build import build_model
import engine.comm as comm
from engine.log.logger import setup_logger
import torch
from codes.data.build import build_dataset
import os
import tqdm
from codes.data.fn.collate_fn import fromlist
from codes.train.train_fn import save_image


class InferenceGt:
    criterion_pixel_wise = torch.nn.MSELoss()

    def __init__(self, cfg, is_tif=True):
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)

        self.model = build_model(cfg)
        self.model.disable_train()

        self.check_pointer = CheckPointStateDict(save_dir='', save_to_disk=False)

        self.device = cfg.MODEL.DEVICE
        self.model_path = cfg.MODEL.WEIGHTS

        if is_tif:
            self.unnormalizing_value = 65535
        else:
            self.unnormalizing_value = 255

        return

    def loop(self, cfg, skip=False, special_name=None):
        if special_name is not None:
            assert (isinstance(special_name, list) or isinstance(special_name, tuple)) and len(special_name) > 0

        test_dataset = build_dataset(cfg, mode='test')
        output = cfg.OUTPUT_DIR
        logging.getLogger(__name__).info('create dataset {}, load {} test data'.format(cfg.DATALOADER.DATASET, len(test_dataset)))

        img_format = 'jpg' if self.unnormalizing_value == 255 else 'tif'

        skin_name = list()
        if skip:
            skin_name = [name for name in os.listdir(output) if name.lower().endswith('jpg')]
        for index in tqdm.tqdm(range(len(test_dataset))):
            data = fromlist([test_dataset[index]])
            input_name = data['name'][0]
            if input_name.endswith('tif') and img_format != 'tif':
                input_name = '{}.{}'.format(input_name[:input_name.rfind('.tif')], img_format)

            if special_name is not None and input_name not in special_name:
                continue
            if input_name in skin_name:
                continue

            real = data["input"].to(self.device)

            with torch.no_grad():
                enhance_img, cure_param = self.model.generator(real)

            if 'expert' in data.keys():
                img_sample = torch.cat((real, enhance_img, data["expert"].to(self.device)), -1)
            else:
                img_sample = torch.cat((real, enhance_img), -1)

            save_image(img_sample, '{}/{}'.format(output, input_name), unnormalizing_value=self.unnormalizing_value, nrow=1, normalize=False)

        return

    def resume_or_load(self):
        model_state_dict, _ = self.check_pointer.resume_or_load(self.model_path, resume=False)

        self.model.load_state_dict(model_state_dict)
        logging.getLogger(__name__).info('load model from {}'.format(self.model_path))
        return



