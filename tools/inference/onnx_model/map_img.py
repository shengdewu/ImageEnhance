import torch
from torch.autograd import Variable


def _map_curl_onnx(img, C, channel_in, channel_out, device='cuda'):
    """Applies a peicewise linear curve defined by a set of knot points to
      an image channel

      :param img: image to be adjusted
      :param C: predicted knot points of curve
      :returns: adjusted image
      :rtype: Tensor

      """
    slope = Variable(torch.zeros((C.shape[0], C.shape[1] - 1))).to(device)
    curve_steps = C.shape[1] - 1

    '''
    Compute the slope of the line segments
    '''
    for i in range(0, curve_steps):
        slope[:, i] = C[:, i + 1] - C[:, i]

    '''
    Use predicted line segments to compute scaling factors for the channel
    '''

    new_img = img.clone()
    new_img[channel_in, :, :] = img[channel_in, :, :] * curve_steps
    scale = torch.full(size=(img.shape[1], img.shape[2]), fill_value=C[0, 0].data, device=device, dtype=img.dtype)
    for i in range(0, slope[0].shape[0] - 1):
        # scale += slope[0][i] * (img[channel_in, :, :] * curve_steps - i)
        scale += slope[0][i] * (new_img[channel_in, :, :] - i)

    new_img[channel_out, :, :] = img[channel_out, :, :] * scale

    return torch.clamp(new_img, 0, 1)


def map_rgb_onnx(img, r, g, b):
    """Adjust the RGB channels of a RGB image using learnt curves

    :param img: image to be adjusted
    :param S: predicted parameters of piecewise linear curves
    :returns: adjust image, regularisation term
    :rtype: Tensor, float

    """
    device = img.device

    '''
    Apply the curve to the R channel 
    '''
    img = _apply_curve_onnx(img, r, channel_in=0, channel_out=0, device=device)

    '''
    Apply the curve to the G channel 
    '''
    img = _apply_curve_onnx(img, g, channel_in=1, channel_out=1, device=device)

    '''
    Apply the curve to the B channel 
    '''
    img = _apply_curve_onnx(img, b, channel_in=2, channel_out=2, device=device)

    return torch.clamp(img, 0, 1.0)