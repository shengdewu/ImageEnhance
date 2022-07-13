import torch

"""
Photometric/digital ITU BT.709:
Y = 0.2126 R + 0.7152 G + 0.0722 B


Digital ITU BT.601 (gives more weight to the R and B components):
Y = 0.299 R + 0.587 G + 0.114 B
"""


def rgb2luma_bt601(image: torch.Tensor):
    '''
    :param image: c, h, w / rgb
    :return:
    '''
    return 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]


def rgb2luma_bt709(image: torch.Tensor):
    '''
    :param image: c, h, w / rgb
    :return:
    '''
    return 0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2]


def rgb2luma_bt601_nchw(image: torch.Tensor):
    '''
    :param image: n, c, h, w / rgb
    :return:
    '''
    return 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
