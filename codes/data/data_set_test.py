import os
import numpy as np
from torch.utils.data import Dataset
from codes.data.build import BUILD_DATASET_REGISTRY
import cv2
from codes.data.fn import tif_opt, augmentation


@BUILD_DATASET_REGISTRY.register()
class ImageDatasetTest(Dataset):
    def __init__(self, cfg, model=''):
        self.root = cfg.DATALOADER.DATA_PATH
        self.scale_factor = cfg.INPUT.DOWN_FACTOR

        self.test_input_files = [os.path.join(self.root, name) for name in os.listdir(self.root)]

        test_max_nums = cfg.DATALOADER.get('XT_TEST_MAX_NUMS', len(self.test_input_files))
        if 0 < test_max_nums < len(self.test_input_files):
            index = [i for i in range(len(self.test_input_files))]
            index = np.random.choice(index, test_max_nums, replace=False)
            self.test_input_files = [self.test_input_files[i] for i in index]
        return

    def __scale__(self, img):
        c, h, w = img.shape
        h = (h // self.scale_factor) * self.scale_factor
        w = (w // self.scale_factor) * self.scale_factor
        return img[:, :h, :w]

    def __getitem__(self, index):

        input_file = self.test_input_files[index % len(self.test_input_files)]
        img_name = os.path.split(input_file)[1]

        img_input = cv2.cvtColor(cv2.imread(input_file, -1), cv2.COLOR_BGR2RGB)

        img_input = tif_opt.to_tensor(img_input)

        return {'input': self.__scale__(img_input), 'name': img_name}

    def __len__(self):
        return len(self.test_input_files)

    def get_item(self, index, skin_name, special_name=None, img_format='jpg'):

        input_file = self.test_input_files[index % len(self.test_input_files)]
        img_name = os.path.split(input_file)[1]

        if img_name.endswith('tif') and img_format != 'tif':
            img_name = '{}.{}'.format(img_name[:img_name.rfind('.tif')], img_format)

        if special_name is not None and img_name not in special_name:
            return {"input": None, "name": img_name}

        if img_name in skin_name:
            return {"input": None, "name": img_name}

        img_rgb = cv2.cvtColor(cv2.imread(input_file, -1), cv2.COLOR_BGR2RGB)
        img_input = tif_opt.to_tensor(img_rgb)

        return {'input': self.__scale__(img_input), 'name': img_name}
