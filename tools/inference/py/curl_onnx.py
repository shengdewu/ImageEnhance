import torch
import numpy as np
import cv2
import onnx
import onnxruntime
import time
from tools.inference.onnx_model.curl_attention import CurlAttentionNet
from tools.inference.onnx_model.curl_luma import CurlLumaNet
from tools.inference.onnx_model.curl import CurlNet
from engine.checkpoint.checkpoint_state_dict import CheckPointStateDict
import engine.checkpoint.functional as checkpoint_f
from tools.inference.onnx_model import map_img
import os
import tqdm
from codes.train.train_fn import save_image


def _create_curl_net(cfg):
    return CurlNet(kernel_number=cfg.KERNEL_NUMBER, knot_points=cfg.KNOT_POINTS)


def _create_spline_att_net(cfg):
    return CurlLumaNet(ratio=cfg.CA_RATIO, pre_knot_points=cfg.PRE_KNOT_POINTS, kernel_number=cfg.KERNEL_NUMBER, knot_points=cfg.KNOT_POINTS)


def to_onnx(model_path, spline_cfg, onnx_name, input_size, device='cpu', log_name=''):
    model_state_dict, _ = CheckPointStateDict(save_dir='', save_to_disk=False).resume_or_load(model_path, resume=False)
    spline_model = _create_spline_att_net(spline_cfg)
    checkpoint_f.load_model_state_dict(spline_model, model_state_dict['g_model'], log_name=log_name)
    torch.onnx.export(spline_model,
                      torch.zeros(size=input_size, device=device, dtype=torch.float32),
                      onnx_name,
                      # export_params=False,
                      dynamic_axes={'gray_img': {2: 'h', 3: 'w'}},
                      input_names=['gray_img'],
                      output_names=['L'],
                      opset_version=11)

    model = onnx.load(onnx_name)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))
    return


def load_onnx(onnx_name):
    model = onnx.load(onnx_name)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))

    ort_session = onnxruntime.InferenceSession(onnx_name,
                                               providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
    return ort_session


def normalized(img):
    '''convert numpy.ndarray to torch tensor. \n
        if the image is uint8 , it will be divided by 255;\n
        if the image is uint16 , it will be divided by 65535;\n
        if the image is float , it will not be divided, we suppose your image range should between [0~1] ;\n

    Arguments:
        img {numpy.ndarray} -- image to be converted to tensor.
    '''
    if not isinstance(img, np.ndarray) and (img.ndim in {2, 3}):
        raise TypeError('data should be numpy ndarray. but got {}'.format(type(img)))

    if img.ndim == 2:
        img = img[:, :, None]

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535
    elif img.dtype in [np.float32, np.float64]:
        img = img.astype(np.float32) / 1
    else:
        raise TypeError('{} is not support'.format(img.dtype))

    return img


def py_onnx_run(down_factor, in_path, out_path, ort_session):

    os.makedirs(out_path, exist_ok=True)

    cal_cure_time = list()
    app_cure_time = list()

    for name in tqdm.tqdm(os.listdir(in_path)):
        img_name = '{}.jpg'.format(name[:name.rfind('.tif')])
        if os.path.exists(os.path.join(out_path, img_name)):
            continue
        img_rgb = cv2.cvtColor(cv2.imread(os.path.join(in_path, name), -1), cv2.COLOR_BGR2RGB)
        img_input = normalized(img_rgb)
        if down_factor > 1 and down_factor % 2 == 0:
            h, w, c = img_input.shape
            h = (h // down_factor) * down_factor
            w = (w // down_factor) * down_factor
            img_input = img_input[:h, :w, :]
            img_input = cv2.resize(img_input, (w // down_factor, h // down_factor), cv2.INTER_CUBIC)
        stime = time.time()
        outputs = ort_session.run(None, {'input_img': img_input.transpose((2, 0, 1))[np.newaxis, :]})
        cal_cure_time.append(time.time()-stime)
        torch_img = torch.from_numpy(normalized(img_rgb).transpose((2, 0, 1)))

        stime = time.time()
        enhance_img = map_img.map_rgb_onnx(torch_img, torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1]), torch.from_numpy(outputs[2]))
        app_cure_time.append(time.time()-stime)
        save_image(torch.cat((torch_img, enhance_img), -1), os.path.join(out_path, img_name), nrow=1, normalize=False)

    if len(cal_cure_time) > 0:
        print('total {} / cal_cure:{}-apply_cure:{}'.format(len(cal_cure_time), sum(cal_cure_time)*1.0/len(cal_cure_time), sum(app_cure_time)*1.0/len(app_cure_time)))
