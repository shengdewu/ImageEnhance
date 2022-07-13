import torch
from torch.autograd import Variable


class CureApply:
    def __init__(self):
        return

    @staticmethod
    def _apply_curve(img, C, slope_sqr_diff, channel_in, channel_out, clamp=False, device='cuda'):
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
        Compute the squared difference between slopes
        '''
        for i in range(0, slope.shape[1] - 1):
            slope_sqr_diff += torch.sum((slope[:, i + 1] - slope[:, i]) * (slope[:, i + 1] - slope[:, i]))

        '''
        Use predicted line segments to compute scaling factors for the channel
        '''

        new_img = img.clone()

        batch_size = img.shape[0]
        for batch in range(batch_size):
            scale = torch.full(size=(img.shape[2], img.shape[3]), fill_value=C[batch, 0].data, device=device, dtype=img.dtype)
            for i in range(0, slope[batch].shape[0] - 1):
                if clamp:
                    scale += slope[batch][i] * (torch.clamp(img[batch, channel_in, :, :] * curve_steps - i, 0, 1))
                else:
                    scale += slope[batch][i] * (img[batch, channel_in, :, :] * curve_steps - i)

            new_img[batch, channel_out, :, :] = img[batch, channel_out, :, :] * scale

        return torch.clamp(new_img, 0, 1), slope_sqr_diff

    @staticmethod
    def _apply_luma_curve(img, gray, C, slope_sqr_diff, clamp=False, device='cuda'):
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
        Compute the squared difference between slopes
        '''
        for i in range(0, slope.shape[1] - 1):
            slope_sqr_diff += torch.sum((slope[:, i + 1] - slope[:, i]) * (slope[:, i + 1] - slope[:, i]))

        '''
        Use predicted line segments to compute scaling factors for the channel
        '''

        new_img = img.clone()
        batch_size = img.shape[0]
        for batch in range(batch_size):
            scale = torch.full(size=(gray.shape[2], gray.shape[3]), fill_value=C[batch, 0].data, device=device, dtype=gray.dtype)
            for i in range(0, slope[batch].shape[0] - 1):
                if clamp:
                    scale += slope[batch][i] * (torch.clamp(gray[batch, 0, :, :] * curve_steps - i, 0, 1))
                else:
                    scale += slope[batch][i] * (gray[batch, 0, :, :] * curve_steps - i)

            new_img[batch, 0, :, :] = img[batch, 0, :, :] * scale
            new_img[batch, 1, :, :] = img[batch, 1, :, :] * scale
            new_img[batch, 2, :, :] = img[batch, 2, :, :] * scale
        return torch.clamp(new_img, 0, 1), slope_sqr_diff

    @staticmethod
    def adjust_rgb(img, R):
        """Adjust the RGB channels of a RGB image using learnt curves

        :param img: image to be adjusted
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        device = img.device

        img = img.contiguous()

        '''
        Extract the parameters of the three curves
        '''
        batch_size, channels = R.shape
        per_channels = int(channels / 3)
        R1 = torch.exp(R[:, 0:per_channels])
        R2 = torch.exp(R[:, per_channels: per_channels * 2])
        R3 = torch.exp(R[:, per_channels * 2:per_channels * 3])

        '''
        Apply the curve to the R channel 
        '''
        slope_sqr_diff = Variable(torch.zeros(1) * 0.0).to(device)

        img, slope_sqr_diff = CureApply._apply_curve(img, R1, slope_sqr_diff, channel_in=0, channel_out=0, device=device)

        '''
        Apply the curve to the G channel 
        '''
        img, slope_sqr_diff = CureApply._apply_curve(img, R2, slope_sqr_diff, channel_in=1, channel_out=1, device=device)

        '''
        Apply the curve to the B channel 
        '''
        img, slope_sqr_diff = CureApply._apply_curve(img, R3, slope_sqr_diff, channel_in=2, channel_out=2, device=device)

        return torch.clamp(img, 0, 1.0).contiguous(), slope_sqr_diff

    @staticmethod
    def adjust_luma(img, gray, l):
        """Adjust the luma of a image using learnt curves

        :param img: image to be adjusted
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        device = img.device

        img = img.contiguous()

        '''
        Extract the parameters of the three curves
        '''
        r = torch.exp(l)

        '''
        Apply the curve to the R channel 
        '''
        slope_sqr_diff = Variable(torch.zeros(1) * 0.0).to(device)

        img, slope_sqr_diff = CureApply._apply_luma_curve(img, gray, r, slope_sqr_diff, device=device)
        return torch.clamp(img, 0, 1.0).contiguous(), slope_sqr_diff

    @staticmethod
    def adjust_hsv(img, S):
        """Adjust the HSV channels of a HSV image using learnt curves

        :param img: image to be adjusted
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        device = img.device

        img = img.contiguous()

        batch_size, channels = S.shape
        per_channels = int(channels / 4)

        S1 = torch.exp(S[:, 0:per_channels])
        S2 = torch.exp(S[:, per_channels:per_channels * 2])
        S3 = torch.exp(S[:, per_channels * 2:per_channels * 3])
        S4 = torch.exp(S[:, per_channels * 3:per_channels * 4])

        slope_sqr_diff = Variable(torch.zeros(1) * 0.0).to(device)

        '''
        Adjust Hue channel based on Hue using the predicted curve
        '''
        img, slope_sqr_diff = CureApply._apply_curve(img, S1, slope_sqr_diff, channel_in=0, channel_out=0, device=device)
        '''
        Adjust Saturation channel based on Hue using the predicted curve
        '''
        img, slope_sqr_diff = CureApply._apply_curve(img, S2, slope_sqr_diff, channel_in=0, channel_out=1, device=device)

        '''
        Adjust Saturation channel based on Saturation using the predicted curve
        '''
        img, slope_sqr_diff = CureApply._apply_curve(img, S3, slope_sqr_diff, channel_in=1, channel_out=1, device=device)

        '''
        Adjust Value channel based on Value using the predicted curve
        '''
        img, slope_sqr_diff = CureApply._apply_curve(img, S4, slope_sqr_diff, channel_in=2, channel_out=2, device=device)

        return torch.clamp(img, 0, 1.0).contiguous(), slope_sqr_diff

    @staticmethod
    def adjust_lab(img, L):
        """Adjusts the image in LAB space using the predicted curves

        :param img: Image tensor
        :param L: Predicited curve parameters for LAB channels
        :returns: adjust image, and regularisation parameter
        :rtype: Tensor, float

        """
        device = img.device

        img = img.contiguous()

        batch_size, channels = L.shape
        per_channels = int(channels / 3)
        '''
        Extract predicted parameters for each L,a,b curve
        '''
        L1 = torch.exp(L[:, 0:per_channels])
        L2 = torch.exp(L[:, per_channels:per_channels * 2])
        L3 = torch.exp(L[:, per_channels * 2:per_channels * 3])

        slope_sqr_diff = Variable(torch.zeros(1) * 0.0).to(device)

        '''
        Apply the curve to the L channel 
        '''
        img, slope_sqr_diff = CureApply._apply_curve(img, L1, slope_sqr_diff, channel_in=0, channel_out=0, device=device)

        '''
        Now do the same for the a channel
        '''
        img, slope_sqr_diff = CureApply._apply_curve(img, L2, slope_sqr_diff, channel_in=1, channel_out=1, device=device)

        '''
        Now do the same for the b channel
        '''
        img, slope_sqr_diff = CureApply._apply_curve(img, L3, slope_sqr_diff, channel_in=2, channel_out=2, device=device)

        return torch.clamp(img, 0, 1.0).contiguous(), slope_sqr_diff

    @staticmethod
    def _apply_curve_onnx(img, C, channel_in, channel_out, device='cuda'):
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

    @staticmethod
    def adjust_rgb_onnx(img, r, g, b):
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
        img = CureApply._apply_curve_onnx(img, r, channel_in=0, channel_out=0, device=device)

        '''
        Apply the curve to the G channel 
        '''
        img = CureApply._apply_curve_onnx(img, g, channel_in=1, channel_out=1, device=device)

        '''
        Apply the curve to the B channel 
        '''
        img = CureApply._apply_curve_onnx(img, b, channel_in=2, channel_out=2, device=device)

        return torch.clamp(img, 0, 1.0)