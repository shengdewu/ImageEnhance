import numpy as np


def clip(v, min_v, max_v):
    v = v if v > min_v else min_v
    return v if v < max_v else max_v


def get_luminance(img: np.ndarray):
    """
    img b g r
    """
    return 0.2126 * img[2] + 0.7152 * img[1] + 0.0722 * img[0]
