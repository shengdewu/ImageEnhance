from fvcore.common.registry import Registry

BUILD_NETWORK_REGISTRY = Registry('NETWORK')
BUILD_NETWORK_REGISTRY.__doc__ = """
BUILD_NETWORK_REGISTRY
"""


def build_generator(cfg):
    return BUILD_NETWORK_REGISTRY.get(cfg.MODEL.NETWORK.ARCH)(cfg)