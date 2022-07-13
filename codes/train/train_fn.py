import torch
import math
import numpy as np
import cv2
import torchvision


@torch.no_grad()
def calculate_psnr(trainer, dataloader, device, criterion_pixel_wise, unnormalizing_value=255):
    trainer.disable_train()
    avg_psnr = 0
    for i, batch in enumerate(dataloader):
        img_input = batch["input"].to(device)
        expert = batch["expert"].to(device)
        expert_fake = trainer.generator(img_input)
        if isinstance(expert_fake, list) or isinstance(expert_fake, tuple):
            expert_fake = expert_fake[0]
        expert_fake = torch.round(expert_fake * unnormalizing_value)
        expert = torch.round(expert * unnormalizing_value)
        mse = criterion_pixel_wise(expert_fake, expert)
        mse = torch.clip(mse, 0.00000001, 4294967296.0)
        psnr = 10.0 * math.log10(float(unnormalizing_value) * unnormalizing_value / mse.item())
        avg_psnr += psnr

    return avg_psnr / len(dataloader)


@torch.no_grad()
def visualize_result(trainer, dataloader, device, save_path, criterion_pixel_wise, unnormalizing_value=255):
    trainer.disable_train()
    img_format = 'jpg' if unnormalizing_value == 255 else 'tif'
    for i, batch in enumerate(dataloader):
        img_input = batch["input"].to(device)
        expert = batch["expert"].to(device)
        expert_fake, cure = trainer.generator(img_input)
        img_sample = torch.cat((img_input.data, expert_fake.data, expert.data), -1)
        expert_fake = torch.round(expert_fake * unnormalizing_value)
        expert = torch.round(expert * unnormalizing_value)
        mse = criterion_pixel_wise(expert_fake, expert)
        mse = torch.clip(mse, 0.00000001, 4294967296.0)
        psnr = 10.0 * math.log10(float(unnormalizing_value) * unnormalizing_value / mse.item())
        save_image(img_sample, '{}/{}-{}.{}'.format(save_path, i, str(psnr)[:5], img_format), unnormalizing_value=unnormalizing_value, nrow=1, normalize=False)
    return


@torch.no_grad()
def save_image(tensor, fp, unnormalizing_value=255, **kwargs):
    fmt = np.uint8 if unnormalizing_value == 255 else np.uint16
    grid = torchvision.utils.make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, unnormalizing_value] to round to nearest integer
    ndarr = grid.mul(unnormalizing_value).add_(0.5).clamp_(0, unnormalizing_value).permute(1, 2, 0).to('cpu').numpy().astype(fmt)
    cv2.imwrite(fp, ndarr[:, :, ::-1])
    return
