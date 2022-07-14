from engine.config.parser import default_argument_parser
from engine.config import get_cfg
from engine.log.logger import setup_logger
from fvcore.common.config import CfgNode
import engine.comm as comm
import codes.model as key_model
import inspect
import logging
import re
import os

from .old_config_new_config import convert_old


def merge_config(use_model_config=False):
    args = default_argument_parser().parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    weights = cfg.MODEL.WEIGHTS

    setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=cfg.OUTPUT_LOG_NAME)

    model_path_root = weights[:weights.rfind('/')]
    # if weights != '':
    #     patterns = inspect.getmembers(key_model, inspect.isclass)
    #     for pattern in patterns:
    #         key = r'{}_.*\.pth'.format(pattern[0])
    #         pattern_obj = re.compile(key)
    #         dg = pattern_obj.search(weights)
    #         if dg is not None:
    #             assert weights.find(pattern[0]) != -1
    #             model_path_root = weights[: dg.span()[0]]
    #             logging.getLogger(cfg.OUTPUT_LOG_NAME).info('from {} found {}'.format(weights, pattern[0]))
    #             break

    train_config = os.path.join(model_path_root, 'config.yaml')
    if use_model_config and os.path.exists(train_config):
        logging.getLogger(cfg.OUTPUT_LOG_NAME).info('use {}'.format(train_config))

        f = open(train_config, mode='r')
        hcfg = CfgNode().load_cfg(f)
        f.close()

        # hcfg = convert_old(hcfg)

        vgg_path = ''
        if cfg.MODEL.get('VGG', None) is not None:
            vgg_path = cfg.MODEL.VGG.PATH
        weights = cfg.MODEL.WEIGHTS
        device = cfg.MODEL.DEVICE
        down_factor = cfg.INPUT.DOWN_FACTOR
        cfg.SOLVER = hcfg.SOLVER
        cfg.MODEL = hcfg.MODEL

        if cfg.MODEL.get('VGG', None) is not None:
            cfg.MODEL.VGG.PATH = vgg_path
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.DEVICE = device
        cfg.INPUT.DOWN_FACTOR = down_factor

    cfg.freeze()
    return cfg
