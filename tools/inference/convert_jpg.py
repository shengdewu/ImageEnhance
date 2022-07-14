import os
import cv2
import numpy as np


def convert2jpg(path, special_name=None):
    base_names = [name for name in os.listdir(path) if name.find('lut') == -1 and name.lower().endswith('tif')]
    skip_names = [name for name in os.listdir(path) if name.lower().endswith('jpg')]
    for name in base_names:
        if special_name is not None and name not in special_name:
            continue
        new_name = '{}.jpg'.format(name[:name.rfind('.tif')])
        if new_name in skip_names:
            continue
        img = cv2.imread(os.path.join(path, name), cv2.IMREAD_UNCHANGED)
        img = np.clip((img / 65535) * 255 + 0.5, 0, 255).astype(np.uint8)
        cv2.imwrite('{}/{}'.format(path, new_name), img)