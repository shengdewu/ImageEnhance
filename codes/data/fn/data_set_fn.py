import os
import random


def search_files(root, txt, skip_name):
    file = open(os.path.join(root, txt), 'r')
    sorted_input_files = sorted([name.strip('\n') for name in file.readlines() if name.find('input,gt') == -1])
    file.close()
    file_names = list()
    for name in sorted_input_files:
        name = name.split(',')
        assert len(name) == 2
        if name[0] in skip_name or name[1] in skip_name:
            continue

        b_input = os.path.join(root, name[0])
        b_expert = os.path.join(root, name[1])
        if not os.path.exists(b_input) or not os.path.exists(b_expert):
            continue
        file_names.append((b_input, b_expert))
    random.shuffle(file_names)
    return file_names
