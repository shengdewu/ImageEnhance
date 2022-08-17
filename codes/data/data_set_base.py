import os
import numpy as np
from torch.utils.data import Dataset
from codes.data.fn.data_set_fn import search_files
from .fn.resize_fn import MaxEdgeResize
import logging


class ImageDataSet(Dataset):
    def __init__(self, cfg, mode):
        root = cfg.DATALOADER.DATA_PATH
        self.mode = mode

        self.skip_name = list()
        if os.path.exists(os.path.join(root, 'skip.txt')):
            file = open(os.path.join(root, 'skip.txt'), 'r')
            self.skip_name = [name.strip('\n') for name in file.readlines()]
            file.close()

        # set_input_files input_name gt_name
        self.set_input_files = search_files(root, cfg.DATALOADER.XT_TRAIN_INPUT_TXT, self.skip_name)
        self.set_label_files = search_files(root, cfg.DATALOADER.XT_TRAIN_LABEL_TXT, self.skip_name)
        self.test_input_files = search_files(root, cfg.DATALOADER.XT_TEST_TXT, self.skip_name)

        test_max_nums = cfg.DATALOADER.get('XT_TEST_MAX_NUMS', len(self.test_input_files))
        if 0 < test_max_nums < len(self.test_input_files):
            index = [i for i in range(len(self.test_input_files))]
            index = np.random.choice(index, test_max_nums, replace=False)
            self.test_input_files = [self.test_input_files[i] for i in index]

        self.max_resize = None
        if cfg.INPUT.get('MAX_RESIZE', 0) > 0:
            self.max_resize = MaxEdgeResize(cfg.INPUT.MAX_RESIZE)

        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('create {}\nMAX_RESIZE: {}'.format(self.__class__, self.max_resize))
        return

    def __getitem__(self, index):
        raise NotImplemented('the loop must be implement')

    def __len__(self):
        if self.mode == 'train':
            return len(self.set_input_files)
        else:
            return len(self.test_input_files)