import torchvision.transforms.functional as F
import engine.transforms.functional as ETF
import numpy as np
import logging
import torch
from .high_light import tone_high_light
from .kelvin_rgb_table import KelvinToRGBTable
import random


def range_float(start, end, step, exclude):
    assert start <= end
    if step >= 1 and int(step * 10) == (step // 1) * 10:
        return [i for i in np.arange(int(start), int(end + 1), step=int(step)) if i != int(exclude)]
    else:
        base = pow(10, len(str(step).split('.')[1]))
        return [i / base for i in np.arange(int(start * base), int(end * base + 1), step=int(step * base)) if i != int(exclude * base)]


class Saturation:
    """
    调整图像的饱和度, 0.0 黑白图
    """

    def __init__(self, f_min=0.8, f_max=1.6, step=0.05, log_name=''):
        self.factor = range_float(f_min, f_max, step=step, exclude=1.0)
        logging.getLogger(log_name).info('{}/{}'.format(self.__class__, self.factor))
        return

    def __call__(self, image, factor=None):
        random_factor = random.choice(self.factor) if factor is None else factor
        return F.adjust_saturation(image, random_factor)

    def __str__(self):
        return 'Saturation'


class Brightness:
    def __init__(self, f_min=0.6, f_max=1.1, step=0.05, log_name=''):
        self.factor = range_float(f_min, f_max, step=step, exclude=1.0)
        logging.getLogger(log_name).info('{}/{}'.format(self.__class__, self.factor))
        return

    def __call__(self, image, factor=None):
        random_factor = random.choice(self.factor) if factor is None else factor
        return F.adjust_brightness(image, random_factor)

    def __str__(self):
        return 'Brightness'


class Contrast:
    """
    调整图像的对比度
    """

    def __init__(self, f_min=0.6, f_max=1.2, step=0.05, log_name=''):
        self.factor = range_float(f_min, f_max, step=step, exclude=1.0)
        logging.getLogger(log_name).info('{}/{}'.format(self.__class__, self.factor))
        return

    def __call__(self, image, factor=None):
        random_factor = random.choice(self.factor) if factor is None else factor
        return F.adjust_contrast(image, random_factor)

    def __str__(self):
        return 'Contrast'


class HighLight:
    def __init__(self, f_min=30, f_max=50, step=10, log_name=''):
        self.random_factor = range_float(start=f_min, end=f_max, step=step, exclude=f_max + 1)
        logging.getLogger(log_name).info('{}/{}'.format(self.__class__, self.random_factor))
        return

    def __call__(self, image, factor=None):
        light = random.choice(self.random_factor) if factor is None else factor
        return tone_high_light(image, light)

    def __str__(self):
        return 'HighLight'


class Hue:
    def __init__(self, f_min=-0.5, f_max=0.5, step=0.05, log_name=''):
        f_min = f_min if f_min > -0.5 else -0.5
        f_max = f_max if f_max < 0.5 else 0.5
        self.factor = range_float(f_min, f_max, step=step, exclude=0.0)
        logging.getLogger(log_name).info('{}/{}'.format(self.__class__, self.factor))
        return

    def __call__(self, image, factor=None):
        random_factor = random.choice(self.factor) if factor is None else factor
        return F.adjust_hue(image, random_factor)

    def __str__(self):
        return 'Hue'


class ColorTemperature:
    def __init__(self, f_min=2000, f_max=40000, step=600, f_mid=None, mid_step=None, log_name=''):
        self.table = KelvinToRGBTable()
        f_mid = f_max if f_mid is None else f_mid
        self.kelvins = range_float(start=f_min, end=f_mid, step=step, exclude=f_mid + 100)
        if f_max > f_mid:
            self.kelvins.extend(range_float(start=f_mid + mid_step, end=f_max, step=mid_step, exclude=f_max + 600))
        logging.getLogger(log_name).info('{}/{}'.format(self.__class__, self.kelvins))
        return

    def __call__(self, image, factor=None):
        kelvin = random.choice(self.kelvins)
        kelvin = np.array(kelvin, dtype=np.float32)
        rgb_multipliers = self.table.transform_kelvins_to_rgb_multipliers(kelvin)
        rgb_multipliers_chw = (rgb_multipliers[::-1]).reshape((3, 1, 1))
        return image * rgb_multipliers_chw

    def __str__(self):
        return 'ColorTemperature'


class AdaptiveLight:
    def __init__(self, f_min=0.6, f_max=1.1, step=0.05, log_name=''):
        self.factor = range_float(f_min, f_max, step=step, exclude=1.0)
        logging.getLogger(log_name).info('{}/{}'.format(self.__class__, self.factor))
        return

    def __call__(self, image, factor=None):
        random_factor = random.choice(self.factor) if factor is None else factor
        f_min = 1.0
        f_max = random_factor
        if random_factor < 1.0:
            f_min = random_factor
            f_max = 1.0
        return ETF.adjust_brightness_adaptive_inv(image, f_min, f_max)

    def __str__(self):
        return 'AdaptiveLight'


class ColorJitter:

    def __init__(self, cfg, color_prob=0., compose_color_prob=0., log_name=''):

        self.color_augmentation_prob = color_prob
        self.color_augmentation = list()

        self.compose_color_augmentation_prob = compose_color_prob
        self.compose_augmentation = list()

        if cfg.get('BRIGHTNESS', None) is not None and cfg.BRIGHTNESS.ENABLE:
            brightness = Brightness(f_min=cfg.BRIGHTNESS.MIN, f_max=cfg.BRIGHTNESS.MAX, step=cfg.BRIGHTNESS.get('STEP', 0.05), log_name=log_name)
            self.color_augmentation.append(brightness)
            if cfg.BRIGHTNESS.get('COMPOSE', False):
                self.compose_augmentation.append(brightness)
                logging.getLogger(log_name).info("add compose: {}".format(brightness.__class__))

        if cfg.get('SATURATION', None) is not None and cfg.SATURATION.ENABLE:
            saturation = Saturation(f_min=cfg.SATURATION.MIN, f_max=cfg.SATURATION.MAX, step=cfg.SATURATION.get('STEP', 0.05), log_name=log_name)
            self.color_augmentation.append(saturation)
            if cfg.SATURATION.get('COMPOSE', False):
                self.compose_augmentation.append(saturation)
                logging.getLogger(log_name).info("add compose: {}".format(saturation.__class__))

        if cfg.get('CONTRAST', None) is not None and cfg.CONTRAST.ENABLE:
            contrast = Contrast(f_min=cfg.CONTRAST.MIN, f_max=cfg.CONTRAST.MAX, step=cfg.CONTRAST.get('STEP', 0.05), log_name=log_name)
            self.color_augmentation.append(contrast)
            if cfg.CONTRAST.get('COMPOSE', False):
                self.compose_augmentation.append(contrast)
                logging.getLogger(log_name).info("add compose: {}".format(contrast.__class__))

        if cfg.get('HIGH_LIGHT', None) is not None and cfg.HIGH_LIGHT.ENABLED:
            high_light = HighLight(f_min=cfg.HIGH_LIGHT.MIN, f_max=cfg.HIGH_LIGHT.MAX, step=cfg.HIGH_LIGHT.STEP, log_name=log_name)
            self.color_augmentation.append(high_light)
            if cfg.HIGH_LIGHT.get('COMPOSE', False):
                self.compose_augmentation.append(high_light)
                logging.getLogger(log_name).info("add compose: {}".format(high_light.__class__))

        if cfg.get('HUE', None) is not None and cfg.HUE.ENABLE:
            hue = Hue(f_min=cfg.HUE.MIN, f_max=cfg.HUE.MAX, step=cfg.HUE.get('STEP', 0.05), log_name=log_name)
            self.color_augmentation.append(hue)
            if cfg.HUE.get('COMPOSE', False):
                self.compose_augmentation.append(hue)
                logging.getLogger(log_name).info("add compose: {}".format(hue.__class__))

        if cfg.get('COLOR_TEMPERATURE', None) is not None and cfg.COLOR_TEMPERATURE.ENABLED:
            color_temperature = ColorTemperature(f_min=cfg.COLOR_TEMPERATURE.MIN,
                                                 f_mid=cfg.COLOR_TEMPERATURE.MID,
                                                 step=cfg.COLOR_TEMPERATURE.STEP,
                                                 f_max=cfg.COLOR_TEMPERATURE.MAX,
                                                 mid_step=cfg.COLOR_TEMPERATURE.MID_STEP,
                                                 log_name=cfg.OUTPUT_LOG_NAME)
            self.color_augmentation.append(color_temperature)
            if cfg.COLOR_TEMPERATURE.get('COMPOSE', False):
                self.compose_augmentation.append(color_temperature)
                logging.getLogger(log_name).info("add compose: {}".format(color_temperature.__class__))

        if cfg.get('ADAPTIVE_LIGHT', None) is not None and cfg.ADAPTIVE_LIGHT.ENABLED:
            adaptive_light = AdaptiveLight(f_min=cfg.ADAPTIVE_LIGHT.MIN,
                                           f_max=cfg.ADAPTIVE_LIGHT.MAX,
                                           step=cfg.ADAPTIVE_LIGHT.get('STEP', 0.05),
                                           log_name=log_name)
            self.color_augmentation.append(adaptive_light)
            if cfg.ADAPTIVE_LIGHT.get('COMPOSE', False):
                self.compose_augmentation.append(adaptive_light)
                logging.getLogger(log_name).info("add compose: {}".format(adaptive_light.__class__))

        if len(self.compose_augmentation) < 2:
            self.compose_color_augmentation_prob = 0.
        if len(self.color_augmentation) <= 0:
            self.color_augmentation_prob = 0.

        logging.getLogger(log_name).info('{}: color prob:{}, compose color prob:{}'.format(
            self.__class__,
            self.color_augmentation_prob,
            self.compose_color_augmentation_prob))
        return

    def __call__(self, image, factor=None):
        """
        Args:
            image (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        if random.random() < self.color_augmentation_prob:
            aug = random.choice(self.color_augmentation)
            return aug(image, factor=factor)

        if random.random() < self.compose_color_augmentation_prob:
            for aug in self.compose_augmentation:
                image = aug(image, factor=factor)

        return image

    def __str__(self):
        return 'ColorJitter'


class AdaptiveOverExposure:
    def __init__(self, f_min=0.0, f_max=0.1, f_value=2.0, log_name=''):
        self.f_min = f_min
        self.f_max = f_max
        self.f_value = f_value
        logging.getLogger(log_name).info('{}/f_min={};f_max={},f_value={}'.format(self.__class__, self.f_min, self.f_max, self.f_value))

    def __call__(self, img_input, img_expert):
        """
        img_input c, h, w belong to [0, 1.0]
        img_expert c, h, w belong to [0, 1.0]
        """
        expert_gray = float(torch.mean(0.299 * img_expert[0] + 0.587 * img_expert[1] + 0.114 * img_expert[2]))
        input_gray = float(torch.mean(0.299 * img_input[0] + 0.587 * img_input[1] + 0.114 * img_input[2]))

        diff_gray = expert_gray - input_gray

        if self.f_min <= diff_gray <= self.f_max:
            min_value = 1.0 if 1.0 < self.f_value else self.f_value
            max_value = self.f_value if self.f_value > 1.0 else 1.0
            img_input = ETF.adjust_brightness_adaptive(img_input, min_value, max_value)

        return img_input


class TrainGTAugmentation:
    def __init__(self, cfg, log_name=''):
        self.train_brightness_threshold = cfg.BRIGHTNESS.THRESHOLD
        self.train_brightness_max = cfg.BRIGHTNESS.MAX
        self.train_darkness_threshold = cfg.DARKNESS.THRESHOLD
        self.train_darkness_min = cfg.DARKNESS.MIN
        assert self.train_darkness_min <= 1.0 and self.train_brightness_max >= 1.0, 'the DARKNESS.MIN must be smaller than 1.0 and the BRIGHTNESS.MAX must be bigger than 1.0'
        self.train_contrast = cfg.CONTRAST
        self.train_saturation = cfg.SATURATION
        self.train_high = cfg.HIGH_LIGHT
        self.adjust_high = HighLight()
        logging.getLogger(log_name).info('{}/{}'.format(self.__class__, cfg))
        return

    def __call__(self, img_expert):
        img_expert = F.adjust_contrast(img_expert, self.train_contrast)
        img_expert = F.adjust_saturation(img_expert, self.train_saturation)

        expert_gray = float(torch.mean(0.299 * img_expert[0] + 0.587 * img_expert[1] + 0.114 * img_expert[2]))
        if expert_gray < self.train_brightness_threshold:
            if self.train_brightness_max != 1.0:
                img_expert = ETF.adjust_brightness_adaptive(img_expert, 1.0, self.train_brightness_max)
        elif expert_gray > self.train_darkness_threshold:
            if self.train_darkness_min != 1.0:
                img_expert = ETF.adjust_brightness_adaptive(img_expert, self.train_darkness_min, 1.0)

        img_expert = self.adjust_high(img_expert, self.train_high)
        return img_expert
