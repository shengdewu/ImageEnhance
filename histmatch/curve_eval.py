import numpy as np
import util
import curves


class CurveEvaluator:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.src_hist = [0. for i in range(256)]

        sh, sw, c = source.shape
        s = 300 / max(sw, sh)
        w = sw * s
        h = sh * s
        self.img = np.zeros(shape=(h, w))

        s_lum = util.get_luminance(source)
        t_lum = util.get_luminance(target)
        for y in range(h):
            sy = int(y/s)
            for x in range(w):
                sx = int(x/s)
                l = util.clip(s_lum[sy, sx], 0, 255)
                self.img[sy, sx] = util.clip(t_lum[sy, sx], 0, 255) / 255.0
                self.src_hist[l] = self.src_hist[l] + 1
        return

    def __call__(self, curve: list):
        hist = [0. for i in range(256)]
        c = curves.Curve(curve)

        h, w = self.img.shape
        for y in range(h):
            for x in range(w):
                l = util.clip(c.get_val(self.img[y, x]), 0., 1.) * 255.0
                hist[int(l)] = hist[int(l)] + 1

        ret = 0
        for i in range(len(hist)):
            ret += abs(self.src_hist[i] - hist[i])

        return ret * (0.1 if self.__is_curve(curve) else 1)

    @staticmethod
    def __is_curve(curve: list):
        shoulder = -1
        prev = 0.
        for i in range(1, len(curve), 2):
            if shoulder < 0:
                if curve[i] >= curve[i+1] and curve[i] > 0:
                    shoulder = 1
                elif curve[i] > 0:
                    return False
            elif shoulder == 1:
                if curve[i] < curve[i+1]:
                    shoulder = 0
            else:
                if curve[i] >= curve[i+1] and curve[i] < 1:
                    return False
                elif curve[i+1] < prev:
                    return False
                prev = curve[i+1]
        return shoulder >= 0




