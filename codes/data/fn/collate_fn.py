import torch


def pad_img(img_list, max_size):
    batch_shape = [len(img_list)] + list(img_list[0].shape[:-2]) + max_size
    img_batch = img_list[0].new_full(batch_shape, 0.0)
    for img, pad_img in zip(img_list, img_batch):
        sh = int((pad_img.shape[-2] - img.shape[-2]) / 2)
        sw = int((pad_img.shape[-1] - img.shape[-1]) / 2)
        pad_img[..., sh: img.shape[-2] + sh, sw: img.shape[-1] + sw].copy_(img)
    return img_batch


def fromlist(batch_img_list):
    img_input = [data['A_input'] for data in batch_img_list]
    img_expert = [data['A_exptC'] for data in batch_img_list]
    img_ref = None
    if 'B_exptC' in batch_img_list[0].keys():
        img_ref = [data['B_exptC'] for data in batch_img_list]

    image_sizes = [(im.shape[-2], im.shape[-1]) for im in img_input]
    image_sizes += [(im.shape[-2], im.shape[-1]) for im in img_expert]
    if img_ref is not None:
        image_sizes += [(im.shape[-2], im.shape[-1]) for im in img_ref]

    image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
    max_size = torch.stack(image_sizes_tensor).max(0).values.numpy().tolist()

    input_batch = dict()
    input_batch['name'] = [data['name'] for data in batch_img_list]
    # max_size can be a tensor in tracing mode, therefore convert to list
    input_batch['input'] = pad_img(img_input, max_size)
    input_batch['expert'] = pad_img(img_input, max_size)
    if img_ref is not None:
        input_batch['ref'] = pad_img(img_ref, max_size)

    # import numpy as np
    # import PIL.Image
    #
    # for i in range(len(input_batch['input'])):
    #     input = input_batch['input'][i]
    #     expert = input_batch['expert'][i]
    #     name = input_batch['name'][i]
    #
    #     input = PIL.Image.fromarray(np.uint8(np.array(input) * 255.0).transpose([1, 2, 0]))
    #     expert = PIL.Image.fromarray(np.uint8(np.array(expert) * 255.0).transpose([1, 2, 0]))
    #
    #     w, h = input.size
    #     sh_img = PIL.Image.new(input.mode, (w*2, h))
    #     sh_img.paste(input, (0, 0))
    #     sh_img.paste(expert, (w, 0))
    #
    #     sh_img.show(name)
    #     sh_img.close()
    return input_batch