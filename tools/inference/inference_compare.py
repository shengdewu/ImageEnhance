import os
from .load_inference_config import merge_config
from .py.inference_gt import InferenceNoneGt
from .py.inference_gt_old import InferenceNoneGtOld
import torch
from . import compare_tool
from .py import curl_onnx as curl_onnx
from .py import dce_onnx as dce_onnx


def execute_and_compare(dir_names, compare_names, compare_base_path=None, use_onnx=False, onnx_save_path='', input_size=(1, 3, 750, 500)):
    cfg = merge_config(use_model_config=True)

    if use_onnx:
        onnx_tool = curl_onnx
        if cfg.MODEL.ARCH == 'DceModel':
            onnx_tool = dce_onnx
        onnx_tool.to_onnx(cfg.MODEL.WEIGHTS, onnx_save_path, input_size=input_size, log_name=cfg.OUTPUT_LOG_NAME)
        inference_tool = onnx_tool.load_onnx(onnx_save_path)
    else:
        inference_tool = InferenceNoneGtOld(cfg, tif=False)
        inference_tool.resume_or_load()

    compare_cls = compare_tool.CompareRow()
    torch.cuda.empty_cache()
    out_root = cfg.OUTPUT_DIR
    data_path = cfg.DATALOADER.DATA_PATH
    compare_root = out_root[:out_root.rfind('/')]

    for dir_name_t in dir_names:
        dir_name = dir_name_t
        save_dir_name = dir_name_t
        if isinstance(dir_name_t, list):
            dir_name = dir_name_t[1]
            save_dir_name = dir_name_t[0]
        data_cfg = cfg.clone()
        data_cfg.defrost()
        out_path = os.path.join(out_root, dir_name)
        data_cfg.OUTPUT_DIR = out_path
        data_cfg.DATALOADER.DATA_PATH = os.path.join(data_path, dir_name)
        print('inference {}'.format(data_cfg.DATALOADER.DATA_PATH))
        if use_onnx:
            onnx_tool.py_onnx_run(cfg.INPUT.DOWN_FACTOR, data_cfg.DATALOADER.DATA_PATH, out_path, inference_tool, skip=True)
        else:
            inference_tool.loop(data_cfg, skip=True)

        if compare_base_path is not None and len(compare_base_path) > 0:
            base_path = os.path.join(compare_base_path, dir_name)
            # convert2jpg(base_path)
            compare_paths = [os.path.join(compare_root, name, dir_name) for name in compare_names]
        else:
            base_path = os.path.join(compare_root, compare_names[0], dir_name)
            compare_paths = [os.path.join(compare_root, name, dir_name) for name in compare_names[1:]]

        out_path = os.path.join(compare_root, 'compare-{}'.format('-'.join(compare_names)), save_dir_name)
        special_name = [name for name in os.listdir(base_path) if name.lower().endswith('jpg')]
        if len(special_name) == 0:
            print('{} no item'.format(base_path))
            continue
        compare_cls.compare(base_path=base_path, compare_paths=compare_paths, out_path=out_path, skip=True, special_name=special_name)
