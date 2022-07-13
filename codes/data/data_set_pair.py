from codes.data.data_set_base import ImageDataSet
from codes.data.build import BUILD_DATASET_REGISTRY
import os
import cv2
import numpy as np
from codes.data.fn import tif_opt, augmentation


@BUILD_DATASET_REGISTRY.register()
class ImageDatasetPaired(ImageDataSet):
    def __init__(self, cfg, mode='train'):
        super(ImageDatasetPaired, self).__init__(cfg, mode)

        self.set_input_files.extend(self.set_label_files)

        self.color_jitter_train = False
        if cfg.INPUT.get('TRAINING_COLOR_JITTER', None) is not None and cfg.INPUT.TRAINING_COLOR_JITTER.ENABLE and self.mode == 'train':
            self.train_gt_aug = augmentation.TrainGTAugmentation(cfg.INPUT.TRAINING_COLOR_JITTER, cfg.OUTPUT_LOG_NAME)
            self.color_jitter_train = cfg.INPUT.TRAINING_COLOR_JITTER.ENABLE

        self.flip_ration = cfg.INPUT.FLIP.RATION

        self.color_jitter = augmentation.ColorJitter(cfg.INPUT.COLOR_JITTER, cfg.OUTPUT_LOG_NAME)

        self.color_jitter_prob = cfg.INPUT.COLOR_JITTER.PROB

        self.input_over_exposure_enable = cfg.INPUT.INPUT_OVER_EXPOSURE.ENABLED
        if self.input_over_exposure_enable and self.mode == 'train':
            self.input_over_exposure = augmentation.AdaptiveOverExposure(f_min=cfg.INPUT.INPUT_OVER_EXPOSURE.F_MIN,
                                                                         f_max=cfg.INPUT.INPUT_OVER_EXPOSURE.F_MAX,
                                                                         f_value=cfg.INPUT.INPUT_OVER_EXPOSURE.F_VALUE,
                                                                         log_name=cfg.OUTPUT_LOG_NAME)

        return

    def __getitem__(self, index):

        if self.mode == 'train':
            input_file = self.set_input_files[index % len(self.set_input_files)]
            img_name = os.path.split(input_file[0])[-1]
            img_input = cv2.cvtColor(cv2.imread(input_file[0], -1), cv2.COLOR_BGR2RGB)
            img_expert = cv2.cvtColor(cv2.imread(input_file[1], -1), cv2.COLOR_BGR2RGB)
            if np.random.random() < self.flip_ration:
                img_input = tif_opt.vflip(img_input)
                img_expert = tif_opt.vflip(img_expert)
        else:
            input_file = self.test_input_files[index % len(self.test_input_files)]
            img_name = os.path.split(input_file[0])[-1]
            img_input = cv2.cvtColor(cv2.imread(input_file[0], -1), cv2.COLOR_BGR2RGB)
            img_expert = cv2.cvtColor(cv2.imread(input_file[1], -1), cv2.COLOR_BGR2RGB)

        img_input = tif_opt.to_tensor(img_input)
        img_expert = tif_opt.to_tensor(img_expert)

        if self.scale_factor > 1:
            c, h, w = img_input.shape
            h = (h // self.scale_factor) * self.scale_factor
            w = (w // self.scale_factor) * self.scale_factor
            img_input = img_input[:, :h, :w]
            img_expert = img_expert[:, :h, :w]

        if self.mode == 'train':
            if self.input_over_exposure_enable:
                img_input = self.input_over_exposure(img_input, img_expert)

            if np.random.random() < self.color_jitter_prob:
                img_input = self.color_jitter(img_input)

            if self.color_jitter_train:
                img_expert = self.train_gt_aug(img_expert)

        return {'A_input': img_input, 'A_exptC': img_expert, 'name': img_name}
