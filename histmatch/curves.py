import util


class Curve:
    def __init__(self, p: list, kind: str= 'DCT_Spline', poly_pn: int = 1000):
        self.ypp = None

        self.ppn = 65500 if poly_pn > 65500 else poly_pn

        self.hash_size = 1000
        if self.ppn < 500:
            self.hash_size = 100

        if self.ppn < 50:
            self.hash_size = 10

        if len(p) < 3:
            return

        identity = True
        self.N = int((len(p) - 1) / 2)
        self.x = [0. for i in range(self.N)]
        self.y = [0. for i in range(self.N)]
        ix = 1

        for i in range(0, self.N, 1):
            self.x[i] = p[ix]
            ix += 1
            self.y[i] = p[ix]
            ix += 1
            if abs(self.x[i] - self.y[i]) >= 0.000009:
                # the smallest possible difference between self.x and self.y curve point values is ~ 0.00001
                # checking against >= 0.000009 is a bit saver than checking against >= 0.00001
                identity = False
        if self.x[0] != 0.0 and self.x[self.N-1] != 1.0:
            # Special (and very rare) case where all points are on the identity line but
            # not reaching the limits
            identity = False
        if self.x[0] == 0. and self.x[1] == 0.:
            # Avoid crash when first two points are at self.x = 0 (git Issue 2888)
            self.x[1] = 0.01
        if self.x[0] == 1. and self.x[1] == 1.:
            # Avoid crash when first two points are at self.x = 1 (100 in gui) (git Issue 2923)
            self.x[0] = 0.99

        if not identity:
            if kind == 'DCT_Spline' and self.N > 2:
                self.spline_cubic_set()
                self.kind = 'DCT_Spline'
            else:
                self.kind = 'DCT_Linear'
        if identity:
            self.kind = 'DCT_Empty'
            print('DCT_Empty')
        return

    def spline_cubic_set(self):
        u = [0. for i in range(self.N-1)]
        self.ypp = [0. for i in range(self.N)]

        self.ypp[0] = u[0] = 0.
        for i in range(1, self.N-1, 1):
            sig = (self.x[i] - self.x[i - 1]) / (self.x[i + 1] - self.x[i - 1])
            p = sig * self.ypp[i-1] + 2.
            self.ypp[i] = (sig - 1.0) / p
            u[i] = ((self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i]) - (self.y[i] - self.y[i - 1]) / (self.x[i] - self.x[i - 1]))
            u[i] = (6.0 * u[i] / (self.x[i + 1] - self.x[i - 1]) - sig * u[i - 1]) / p

        self.ypp[self.N - 1] = 0.
        for k in range(self.N-2, 0, -1):
            self.ypp[k] = self.ypp[k] * self.ypp[k+1] + u[k]

        del u
        return

    def get_val(self, t: float):
        # if not identity and self.N > 2:
        #     self.spline_cubic_set()
        #     self.kind = 'DCT_Spline'
        # if identity:
        #     self.kind = 'DCT_Empty'
        #     print('DCT_Empty')
        if self.kind == 'DCT_Empty':
            return t
        else:
            if t > self.x[self.N-1]:
                return self.y[self.N-1]
            elif t < self.x[0]:
                return self.y[0]
            k_lo = 0
            k_hi = self.N - 1
            while k_hi > 1 + k_lo:
                k = int((k_hi + k_lo) / 2)
                if self.x[k] > t:
                    k_hi = k
                else:
                    k_lo = k
            h = self.x[k_hi] - self.x[k_lo]
            if self.kind == 'DCT_Linear':
                return self.y[k_lo] + (t - self.x[k_lo]) * (self.y[k_hi] - self.y[k_lo]) / h
            else: # DCT_Spline
                a = (self.x[k_hi] - t) / h
                b = (t - self.x[k_lo]) / h
                r = a * self.y[k_lo] + b * self.y[k_hi] + ((a * a * a - a) * self.ypp[k_lo] + (b * b * b - b) * self.ypp[k_hi]) * (h * h) * 0.1666666666666666666666666666666;
                return util.clip(r, 0.0, 1.0)