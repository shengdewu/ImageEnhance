import os
import random
from PIL import Image


def singal_preprocess(data_path, out_path, txt):
    file = open(os.path.join(data_path, txt), 'r')
    input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
    file.close()
    with open(os.path.join(out_path, txt), mode='w') as w:
        w.write('input,gt\n')

        for name in input_files:
            name = name.split(',')
            assert len(name) == 2
            a_input = os.path.join(data_path, "rt_tif_16bit_540p", name[0])
            a_expert = os.path.join(data_path, "gt_16bit_540p", name[1])
            if not os.path.exists(a_input) or not os.path.exists(a_expert):
                continue

            a_input_img = Image.open(a_input)
            a_expert_img = Image.open(a_expert)

            W, H = a_input_img.size
            W2, H2 = a_expert_img.size

            if abs(W - W2) > 2 or abs(H - H2) > 2:
                continue

            min_w = max(W, W2)
            min_h = max(H, H2)

            a_input_rgb = a_input_img.convert('RGB')
            if a_input_rgb.size != (min_w, min_h):
                a_input_rgb = a_input_rgb.resize((min_w, min_h), Image.BILINEAR)
            a_input_rgb.save(os.path.join(out_path, "rt_tif_16bit_540p", '{}.jpg'.format(name[0][0:name[0].rfind('.tif')])))

            a_expert_rgb = a_expert_img.convert('RGB')
            if a_expert_rgb.size != (min_w, min_h):
                a_expert_rgb = a_expert_rgb.resize((min_w, min_h), Image.BILINEAR)
            a_expert_rgb.save(os.path.join(out_path, "gt_16bit_540p", '{}.jpg'.format(name[1][0:name[1].rfind('.tif')])))

            w.write('{},{}\n'.format('{}.jpg'.format(name[0][0:name[0].rfind('.tif')]), '{}.jpg'.format(name[1][0:name[1].rfind('.tif')])))


def preprocess(data_path, out_path):
    os.makedirs(os.path.join(out_path, 'rt_tif_16bit_540p'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'gt_16bit_540p'), exist_ok=True)

    singal_preprocess(data_path, out_path, 'train_input.txt')
    singal_preprocess(data_path, out_path, 'train_label.txt')
    singal_preprocess(data_path, out_path, 'test.txt')


def singal_match(data_path, txt):
    file = open(os.path.join(data_path, txt), 'r')
    input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
    file.close()

    skip_names = list()
    for name in input_files:
        name = name.split(',')
        assert len(name) == 2
        a_input = os.path.join(data_path, "rt_tif_16bit_540p", name[0])
        a_expert = os.path.join(data_path, "gt_16bit_540p", name[1])
        if not os.path.exists(a_input) or not os.path.exists(a_expert):
            continue

        a_input_img = Image.open(a_input)
        a_expert_img = Image.open(a_expert)

        W, H = a_input_img.size
        W2, H2 = a_expert_img.size

        if abs(W - W2) < 2 and abs(H - H2) < 2:
            continue

        skip_names.append(name[0])
        skip_names.append(name[1])

    print('total select {}'.format(len(skip_names)))
    if len(skip_names) > 0:
        with open(os.path.join(data_path, 'skip.txt'), mode='a+') as w:
            for skip_name in set(skip_names):
                w.write('{}\n'.format(skip_name))


def match(data_path):
    singal_match(data_path, 'train_input.txt')
    singal_match(data_path, 'train_label.txt')
    singal_match(data_path, 'test.txt')


def _label(root_path, pair_name, flag='', test_ratio=0.05):
    random.shuffle(pair_name)
    index = [i for i in range(len(pair_name))]
    test_name_index = random.sample(index, int(test_ratio * len(pair_name)))

    assert len(test_name_index) == len(set(test_name_index)), 'the test name index duplication'

    test_name = list()
    train_name = list()
    for i in index:
        if i in test_name_index:
            test_name.append(pair_name[i])
        else:
            train_name.append(pair_name[i])

    with open(os.path.join(root_path, '{}.test.txt'.format(flag)), mode='w') as t:
        t.write('input,gt\n')
        for name in test_name:
            assert len(name) == 2
            t.write('{},{}\n'.format(name[1], name[0]))

    index = [i for i in range(len(train_name))]
    select_index = random.sample(index, int(0.5 * len(train_name)))

    with open(os.path.join(root_path, '{}.train_input.txt'.format(flag)), mode='w') as ti:
        with open(os.path.join(root_path, '{}.train_label.txt'.format(flag)), mode='w') as tl:
            ti.write('input,gt\n')
            tl.write('input,gt\n')
            for i in index:
                name = train_name[i]
                assert len(name) == 2
                if i in select_index:
                    ti.write('{},{}\n'.format(name[1], name[0]))
                else:
                    tl.write('{},{}\n'.format(name[1], name[0]))


def _search_rt(rt_name, proportion=1.):
    with open(os.path.join(rt_name), mode='r') as igt:
        rt_inputs = [r.strip('\n') for r in igt.readlines()]
    if 1. > proportion > 0.:
        rt_inputs = random.sample(rt_inputs, int(proportion * len(rt_inputs)))
    return rt_inputs


def create_train_label(root_path):
    gts = _search_rt(os.path.join(root_path, 'gt_16bit_540p.txt'))
    aug_inputs = dict([(name.split('/')[1], name) for name in _search_rt(os.path.join(root_path, 'rt_tif_aug_16bit_540p.txt'))])
    no_aug_inputs = dict([(name.split('/')[1], name) for name in _search_rt(os.path.join(root_path, 'rt_tif_16bit_540p.txt'))])

    aug_inputs.update(no_aug_inputs)

    select_name = list()
    for gt in gts:
        name = gt.split('/')[1]
        zero = '{}_0{}'.format(name[:-4], name[-4:])
        one = '{}_1{}'.format(name[:-4], name[-4:])
        two = '{}_2{}'.format(name[:-4], name[-4:])
        if name not in aug_inputs.keys() or zero not in aug_inputs.keys() or one not in aug_inputs.keys() or two not in aug_inputs.keys():
            print(gt)
            continue
        select_name.append((gt, aug_inputs[name]))
        select_name.append((gt, aug_inputs[zero]))
        select_name.append((gt, aug_inputs[one]))
        select_name.append((gt, aug_inputs[two]))

    _label(root_path, select_name, 'all')

    select_name = list()
    for gt in gts:
        name = gt.split('/')[1]
        if name not in no_aug_inputs.keys():
            print(gt)
            continue
        select_name.append((gt, no_aug_inputs[name]))

    _label(root_path, select_name, 'no_aug')


def _create_dup_dict(path, name_key=None, proportion=1.):
    name_dict = dict()
    for name in _search_rt(os.path.join(path), proportion=proportion):
        if name_key is not None and name.split('/')[1].find(name_key) == -1:
            continue
        key = '{}.{}'.format(name.split('/')[1].split('g')[0], name.split('/')[1].split('.')[-1])
        if name_dict.get(key, None) is None:
            name_dict[key] = list()
        name_dict[key].append(name)
    return name_dict


def create_train_no_aug_label(root_path, max_rts=-1):
    gts = _search_rt(os.path.join(root_path, 'gt_16bit_540p_only_adjust_light.txt'))
    rt_input_dict_list = list()

    if max_rts != 0:
        rts = _search_rt(os.path.join(root_path, 'rt_tif_16bit_540p.txt'))
        if max_rts > 1:
            rts = random.sample(rts, min(max_rts, len(rts)))
        elif max_rts > 0:  # [0.0, 1)
            rts = random.sample(rts, int(max_rts*len(rts)))
        rt_input_dict_list.append(dict([(name.split('/')[1], name) for name in rts]))

    rt_input_dict_list.append(_create_dup_dict(os.path.join(root_path, 'rt_tif_16bit_540p_overexposure.only.light.v3_01.txt'), proportion=0.2))
    rt_input_dict_list.append(_create_dup_dict(os.path.join(root_path, 'rt_tif_16bit_540p.only.light.local.light.v3_01.txt')))

    train_pair = list()
    for ii in gts:
        if not os.path.exists(os.path.join(root_path, ii)):
            continue

        k = ii.split('/')[-1]
        for rt_input_dict in rt_input_dict_list:
            if k in rt_input_dict.keys():
                if isinstance(rt_input_dict[k], list):
                    for v in rt_input_dict[k]:
                        train_pair.append((ii, v))
                else:
                    train_pair.append((ii, rt_input_dict[k]))

    _label(root_path, train_pair, 'only_adjust_light_base.ov.loc')


if __name__ == '__main__':
    create_train_no_aug_label('/mnt/sdb/data.set/xintu.data/enhance.data/xt.image.enhancement.540', -1)