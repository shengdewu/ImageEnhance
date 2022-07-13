from fvcore.common.registry import Registry

BUILD_DISCRIMINATOR_REGISTRY = Registry('DISCRIMINATOR')
BUILD_DISCRIMINATOR_REGISTRY.__doc__ = """
BUILD_DISCRIMINATOR_REGISTRY
"""


def build_discriminator(cfg):
    return BUILD_DISCRIMINATOR_REGISTRY.get(cfg.MODEL.DISCRIMINATOR.ARCH)()