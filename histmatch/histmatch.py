import cv2
import numpy
import os

import numpy as np


class CdfInfo:
    def __init__(self):
        self.cdf = [0 for i in range(256)]
        self.min_val = -1
        self.max_val = -1
        return


class HistMatching:
    def __init__(self):
        return

    @staticmethod
    def get_luminance(img: np.ndarray):
        """
        img b g r
        """
        return 0.2126 * img[2] + 0.7152 * img[1] + 0.0722 * img[0]

    @staticmethod
    def get_cdf(img: np.ndarray):
        cdf = CdfInfo()
        unique, count = np.unique(img, return_counts=True)
        cdf.min_val = np.min(unique)
        cdf.max_val = np.max(unique)
        cdf.cdf = np.cumsum(count)
        return cdf

    @staticmethod
    def find_match(value, cdf: list, j: int):
        if cdf[j] <= value:
            for i in range(j, len(cdf), 1):
                if cdf[i] == value:
                    return i
                elif cdf[i] > value:
                    return i if cdf[i] - value < cdf[i-1] - value else i-1
            return 255
        else:
            for i in range(j, 0, -1):
                if cdf[i] == value:
                    return i
                elif cdf[i] < value:
                    return i if cdf[i] - value < cdf[i+1] - value else i+1
            return 0

    @staticmethod
    def coord(v: int) -> float:
        return float(v) / 255.0

    def mapping_curve(self, mapping: list, curve: list):
        curve.clear()
        idx = 15
        for i in range(idx, len(mapping), 1):
            if mapping[i] >= i:
                idx = i
                break
        if idx == len(mapping):
            for i in range(1, len(mapping)-1, 1):
                if mapping[i] >= i:
                    idx = i
                    break

        def doit(start: int, stop: int, step: int, addstart: bool, maxdelta:int = 0):
            maxdelta = step * 2 if maxdelta == 0 else maxdelta
            prev = start
            if addstart and mapping[start] >= 0:
                curve.append(self.coord(start))
                curve.append(self.coord(mapping[start]))
            for i in range(start, stop, 1):
                v = mapping[i]
                if v < 0:
                    continue
                change = i > 0 and v != mapping[i-1]
                diff = i - prev
                if (change and abs(diff - step) <= 1) or (diff > maxdelta):
                    curve.append(self.coord(i))
                    curve.append(self.coord(v))
                    prev = i

        curve.append(0.0)
        curve.append(0.0)

        return

    def get_auto_matched_tone_curve(self, target: np.ndarray, source: np.ndarray):
        th, tw, _ = target.shape
        sh, sw, _ = source.shape
        if sh * 5 < th:
            print('histogram matching: the source image is too small {}/{}'.format(sh, sw))
            return

        source = cv2.resize(source, dsize=(tw, th), interpolation=cv2.INTER_NEAREST)

        # source_b = source[:, :, 0]
        # source_g = source[:, :, 1]
        # source_r = source[:, :, 2]
        source_luminance = self.get_luminance(source).astype(np.uint8)
        # target_b = target[:, :, 0]
        # target_g = target[:, :, 1]
        # target_r = target[:, :, 2]
        target_luminance = self.get_luminance(target).astype(np.uint8)

        t_cdf = self.get_cdf(target_luminance)
        s_cdf = self.get_cdf(source_luminance)
        j = 0
        mapping = list()
        for i in range(0, len(t_cdf.cdf), 1):
            j = self.find_match(t_cdf.cdf[i], s_cdf.cdf, j)
            if t_cdf.min_val <= i <= t_cdf.max_val and s_cdf.min_val <= j <= s_cdf.max_val:
                mapping.append(j)
            else:
                mapping.append(-1)

        candidates = list()

        return
