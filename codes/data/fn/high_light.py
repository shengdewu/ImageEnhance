from .rgb2luma import rgb2luma_bt601
import torch


def tone_high_light(image:torch.Tensor, light=50.0, max_val=4):
    '''
    :param image: [0, 1]
    :param light:
    :return:
    '''
    luma = rgb2luma_bt601(image)
    # 确定高光区域
    thresh = luma * luma
    # 取平均值做阈值
    t = torch.mean(thresh)
    mask = torch.where(thresh >= t, 255, 0).short()

    # 边缘平滑过渡
    bright = light / 100.0 / max_val
    mid = 1.0 + max_val * bright
    mid_tmp = (mid-1.0)/t * thresh+1.0
    mid = torch.zeros(size=(1, 1), dtype=torch.float32).to(mask.device) + mid
    mid_rate = torch.where(mask == 255, mid, mid_tmp)
    bright_tmp = (1.0/t * thresh) * bright
    bright = torch.zeros(size=(1, 1), dtype=torch.float32).to(mask.device) + bright
    bright_rate = torch.where(mask == 255, bright, bright_tmp)

    # 获取结果图
    tmp = torch.pow(image, 1.0/mid_rate) * (1.0 / (1 - bright_rate))
    return torch.clip(tmp, 0.0, 1.0)


def tone_high_light_no_luma(image: torch.Tensor, luma: torch.Tensor, light=50.0, max_val=4):
    '''
    :param image: [0, 1]
    :param light:
    :return:
    '''

    # 确定高光区域
    thresh = luma * luma
    # 取平均值做阈值
    t = torch.mean(thresh)
    mask = torch.where(thresh >= t, 255, 0).short()

    # 边缘平滑过渡
    bright = light / 100.0 / max_val
    mid = 1.0 + max_val * bright
    mid_tmp = (mid-1.0)/t * thresh+1.0
    mid = torch.zeros(size=(1, 1), dtype=torch.float32).to(mask.device) + mid
    mid_rate = torch.where(mask == 255, mid, mid_tmp)
    bright_tmp = (1.0/t * thresh) * bright
    bright = torch.zeros(size=(1, 1), dtype=torch.float32).to(mask.device) + bright
    bright_rate = torch.where(mask == 255, bright, bright_tmp)

    # 获取结果图
    tmp = torch.pow(image, 1.0/mid_rate) * (1.0 / (1 - bright_rate))
    return torch.clip(tmp, 0.0, 1.0)


