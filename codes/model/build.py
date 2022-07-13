from fvcore.common.registry import Registry

BUILD_MODEL_REGISTRY = Registry('MODEL')
BUILD_MODEL_REGISTRY.__doc__ = """
BUILD_MODEL_REGISTRY
"""


def build_model(cfg):
    return BUILD_MODEL_REGISTRY.get(cfg.MODEL.ARCH)(cfg)