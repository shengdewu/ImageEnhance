from fvcore.common.registry import Registry

BUILD_DATASET_REGISTRY = Registry('DATASET')
BUILD_DATASET_REGISTRY.__doc__ = """
BUILD_DATASET_REGISTRY
"""


def build_dataset(cfg, mode='train'):
    return BUILD_DATASET_REGISTRY.get(cfg.DATALOADER.DATASET)(cfg, mode)