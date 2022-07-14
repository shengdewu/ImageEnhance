import tools.inference.inference_compare as inference_compare


if __name__ == '__main__':
    # txt_name = '/mnt/sdb/error.collection/lut.test/测试图/test.dir.name.txt'
    txt_name = 'test_dirs/dir_error.txt'
    rhd = open(txt_name, mode='r')
    dir_names = [line.strip('\n').split('#') for line in rhd.readlines()]
    rhd.close()

    compare_name = ['img.spline.att.only_adjust_light_base', 'curl']
    inference_compare.execute_and_compare(dir_names, compare_name)
