import os
import random
import cv2
from codes.data.build import BUILD_DATASET_REGISTRY
from codes.data.data_set_base import ImageDataSet
from codes.data.fn import tif_opt, augmentation


@BUILD_DATASET_REGISTRY.register()
class ImageDataSetTuUnpairedTanh(ImageDataSet):
    def __init__(self, cfg, mode='train'):
        super(ImageDataSetTuUnpairedTanh, self).__init__(cfg, mode)

        self.flip_ration = cfg.INPUT.FLIP.RATION
        self.color_jitter = augmentation.ColorJitter(cfg.INPUT.COLOR_JITTER, cfg.OUTPUT_LOG_NAME)
        self.color_jitter_prob = cfg.INPUT.COLOR_JITTER.PROB

        return

    def __getitem__(self, index):

        if self.mode == 'train':
            input_file = self.set_input_files[index % len(self.set_input_files)]
            img_name = os.path.split(input_file[0])[-1]
            img_input = cv2.cvtColor(cv2.imread(input_file[0], -1), cv2.COLOR_BGR2RGB)
            img_expt_a = cv2.cvtColor(cv2.imread(input_file[1], -1), cv2.COLOR_BGR2RGB)
            seed = random.randint(1, len(self.set_label_files))
            img_expt_b = cv2.cvtColor(cv2.imread(self.set_label_files[(index + seed) % len(self.set_label_files)][1], -1), cv2.COLOR_BGR2RGB)
            if self.max_resize is not None:
                img_input = self.max_resize(img_input)
                img_expt_a = self.max_resize(img_expt_a)
                img_expt_b = self.max_resize(img_expt_b)
            # w = min(img_input.shape[1], img_expt_a.shape[1], img_expt_b.shape[1])
            # h = min(img_input.shape[0], img_expt_a.shape[0], img_expt_b.shape[1])
            #
            # if img_input.shape[:2] != (h, w):
            #     img_input = img_input[0:h, 0:w, :]
            # if img_expt_a.shape[:2] != (h, w):
            #     img_expt_a = img_expt_a[0:h, 0:w, :]
            # if img_expt_b.shape[:2] != (h, w):
            #     img_expt_b = img_expt_b[0:h, 0:w, :]

        else:
            input_file = self.test_input_files[index % len(self.test_input_files)]
            img_name = os.path.split(input_file[0])[-1]
            img_input = cv2.cvtColor(cv2.imread(input_file[0], -1), cv2.COLOR_BGR2RGB)
            img_expt_a = cv2.cvtColor(cv2.imread(input_file[1], -1), cv2.COLOR_BGR2RGB)
            img_expt_b = img_expt_a

        img_input = (tif_opt.to_tensor(img_input) - 0.5) / 0.5
        img_expt_a = (tif_opt.to_tensor(img_expt_a) - 0.5) / 0.5
        img_expt_b = (tif_opt.to_tensor(img_expt_b) - 0.5) / 0.5

        if self.mode == 'train':
            if random.random() < self.color_jitter_prob:
                img_input = self.color_jitter(img_input)

        return {'A_input': img_input, 'A_exptC': img_expt_a, 'B_exptC': img_expt_b, 'name': img_name}

