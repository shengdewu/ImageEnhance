from fvcore.common.config import CfgNode

arch_dict = {
    'CureEnhanceModelWithGT': 'DceModel',
    'SplineAttentionModel': 'CurlModel',
    'SplineRGBL1SSIMLossModel': 'CurlModel',
}


def convert_old(old_cfg):
    new_cfg = old_cfg.clone()
    new_cfg.MODEL.NETWORK = CfgNode(new_allowed=True)

    curl_net = new_cfg.MODEL.get('CURL_NET', None)
    if curl_net is not None:
        new_cfg.MODEL.NETWORK.CURL_NET = CfgNode(new_allowed=True)

        arch = old_cfg.MODEL.ARCH
        if arch_dict.get(arch, None) is None:
            raise NotImplementedError('old arch {} is not implement, new arch only implement DceModel, CurlModel'.format(arch))
        new_cfg.MODEL.ARCH = arch_dict[arch]
        if arch == 'SplineAttentionModel':
            if curl_net.get('LIKE_VGG', False):
                new_cfg.MODEL.NETWORK.ARCH = 'CurlDownNet'
                print('convert old model/network {}/LIKE_VGG to new model/network {}/CurlDownNet'.format(arch, new_cfg.MODEL.ARCH))
                del curl_net['LIKE_VGG']
            elif curl_net.get('LUMA', False):
                new_cfg.MODEL.NETWORK.ARCH = 'CurlLumaNet'
                print('convert old model/network {}/LUMA to new model/network {}/CurlLumaNet'.format(arch, new_cfg.MODEL.ARCH))
                del curl_net['LUMA']
            else:
                new_cfg.MODEL.NETWORK.ARCH = 'CurlAttentionNet'
                print('convert old model/network {}/attention to new model/network {}/CurlAttentionNet'.format(arch, new_cfg.MODEL.ARCH))
        elif arch == 'SplineRGBL1SSIMLossModel':
            new_cfg.MODEL.NETWORK.ARCH = 'CurlNet'
            print('convert old model/network {}/rgb to new model/network {}/CurlNet'.format(arch, new_cfg.MODEL.ARCH))
        else:
            new_cfg.MODEL.NETWORK.ARCH = 'DceNet'
            print('convert old model {} to new model/network {}/DceNet'.format(arch, new_cfg.MODEL.ARCH))

        del new_cfg.MODEL['CURL_NET']
        new_cfg.MODEL.NETWORK.CURL_NET = curl_net

    dce_net = new_cfg.MODEL.get('DCE_NET', None)
    if dce_net is not None:
        new_cfg.MODEL.NETWORK.DCE_NET = dce_net
        del new_cfg.MODEL['DCE_NET']

    new_cfg.SOLVER.OPTIMIZER.BASE_LR = new_cfg.SOLVER.BASE_LR
    del new_cfg.SOLVER['BASE_LR']

    return new_cfg
