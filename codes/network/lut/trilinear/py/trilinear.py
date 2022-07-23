import math
from numba import jit

__all__ = [
    'trilinear_forward',
    'trilinear_backword'
]


@jit(nopython=True)
def trilinear_forward(x, lut, output, w, h, bins):

    for wi in range(w):
        for hi in range(h):
            r = x[0, hi, wi]
            g = x[1, hi, wi]
            b = x[2, hi, wi]

            int_ri = math.floor(r / bins)
            int_gi = math.floor(g / bins)
            int_bi = math.floor(b / bins)

            rem_r = r % bins / bins
            rem_g = g % bins / bins
            rem_b = b % bins / bins

            w000 = (1 - rem_r) * (1 - rem_g) * (1 - rem_b)
            w100 = rem_r * (1 - rem_g) * (1 - rem_b)
            w010 = (1 - rem_r) * rem_g * (1 - rem_b)
            w110 = rem_r * rem_g * (1 - rem_b)
            w001 = (1 - rem_r) * (1 - rem_g) * rem_b
            w101 = rem_r * (1 - rem_g) * rem_b
            w011 = (1 - rem_r) * rem_g * rem_b
            w111 = rem_r * rem_g * rem_b

            id000 = int_bi, int_gi, int_ri
            id100 = int_bi, int_gi, int_ri + 1
            id010 = int_bi, int_gi + 1, int_ri
            id110 = int_bi, int_gi + 1, int_ri + 1
            id001 = int_bi + 1, int_gi, int_ri
            id101 = int_bi + 1, int_gi, int_ri + 1
            id011 = int_bi + 1, int_gi + 1, int_ri
            id111 = int_bi + 1, int_gi + 1, int_ri + 1

            output[0, hi, wi] = w000 * lut[0][id000] + w100 * lut[0][id100] + w010 * lut[0][id010] + w110 * lut[0][id110] + \
                                w001 * lut[0][id001] + w101 * lut[0][id101] + w011 * lut[0][id011] + w111 * lut[0][id111]

            output[1, hi, wi] = w000 * lut[1][id000] + w100 * lut[1][id100] + w010 * lut[1][id010] + w110 * lut[1][id110] + \
                                w001 * lut[1][id001] + w101 * lut[1][id101] + w011 * lut[1][id011] + w111 * lut[1][id111]

            output[2, hi, wi] = w000 * lut[2][id000] + w100 * lut[2][id100] + w010 * lut[2][id010] + w110 * lut[2][id110] + \
                                w001 * lut[2][id001] + w101 * lut[2][id101] + w011 * lut[2][id011] + w111 * lut[2][id111]

    return output


@jit(nopython=True)
def trilinear_backword(x, x_grad, lut_grad, h, w, bins):
    for wi in range(w):
        for hi in range(h):
            r = x[0, hi, wi]
            g = x[1, hi, wi]
            b = x[2, hi, wi]

            int_ri = math.floor(r / bins)
            int_gi = math.floor(g / bins)
            int_bi = math.floor(b / bins)

            rem_r = r % bins / bins
            rem_g = g % bins / bins
            rem_b = b % bins / bins

            w000 = (1 - rem_r) * (1 - rem_g) * (1 - rem_b)
            w100 = rem_r * (1 - rem_g) * (1 - rem_b)
            w010 = (1 - rem_r) * rem_g * (1 - rem_b)
            w110 = rem_r * rem_g * (1 - rem_b)
            w001 = (1 - rem_r) * (1 - rem_g) * rem_b
            w101 = rem_r * (1 - rem_g) * rem_b
            w011 = (1 - rem_r) * rem_g * rem_b
            w111 = rem_r * rem_g * rem_b

            id000 = int_bi, int_gi, int_ri
            id100 = int_bi, int_gi, int_ri + 1
            id010 = int_bi, int_gi + 1, int_ri
            id110 = int_bi, int_gi + 1, int_ri + 1
            id001 = int_bi + 1, int_gi, int_ri
            id101 = int_bi + 1, int_gi, int_ri + 1
            id011 = int_bi + 1, int_gi + 1, int_ri
            id111 = int_bi + 1, int_gi + 1, int_ri + 1

            lut_grad[0][id000] += w000 * x_grad[0, hi, wi]
            lut_grad[0][id100] += w100 * x_grad[0, hi, wi]
            lut_grad[0][id010] += w010 * x_grad[0, hi, wi]
            lut_grad[0][id110] += w110 * x_grad[0, hi, wi]
            lut_grad[0][id001] += w001 * x_grad[0, hi, wi]
            lut_grad[0][id101] += w101 * x_grad[0, hi, wi]
            lut_grad[0][id011] += w011 * x_grad[0, hi, wi]
            lut_grad[0][id111] += w111 * x_grad[0, hi, wi]

            lut_grad[1][id000] += w000 * x_grad[1, hi, wi]
            lut_grad[1][id100] += w100 * x_grad[1, hi, wi]
            lut_grad[1][id010] += w010 * x_grad[1, hi, wi]
            lut_grad[1][id110] += w110 * x_grad[1, hi, wi]
            lut_grad[1][id001] += w001 * x_grad[1, hi, wi]
            lut_grad[1][id101] += w101 * x_grad[1, hi, wi]
            lut_grad[1][id011] += w011 * x_grad[1, hi, wi]
            lut_grad[1][id111] += w111 * x_grad[1, hi, wi]

            lut_grad[2][id000] += w000 * x_grad[2, hi, wi]
            lut_grad[2][id100] += w100 * x_grad[2, hi, wi]
            lut_grad[2][id010] += w010 * x_grad[2, hi, wi]
            lut_grad[2][id110] += w110 * x_grad[2, hi, wi]
            lut_grad[2][id001] += w001 * x_grad[2, hi, wi]
            lut_grad[2][id101] += w101 * x_grad[2, hi, wi]
            lut_grad[2][id011] += w011 * x_grad[2, hi, wi]
            lut_grad[2][id111] += w111 * x_grad[2, hi, wi]

    return




