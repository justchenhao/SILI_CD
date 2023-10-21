import numpy as np
import torch
from torchvision import utils


def str2int(cmd_str: str):
    # usage
    # 1,1 -> [1,1]
    # 0.25,1/0.5/0.25  -> [0.25,[1,0.5,0.25]]
    out_list = []
    items = cmd_str.split(',')
    for item in items:
        if '/' in item:
            scales = item.split('/')
            item = [float(i) for i in scales]
        else:
            item = float(item)
        out_list.append(item)
    return out_list


def modify_state_dicts(state_dict, key_pre='', rm_pre='', rm_key=''):
    """
    提取关键字key开头的keys，并移除其中开头的rm_pre的字符
    Args:
        state_dict:
        key_pre: 提取关键字为key_pre的键，
        rm_pre:  提取的键中rm_pre开头的字符删除，留下后续字符作为新的key
        rm_key: 去除含有rm_key关键字的key
    Returns: out_state_dict

    """
    out_state_dict = {}
    keys = list(state_dict.keys())
    values = list(state_dict.values())
    for key, value in zip(keys, values):
        if rm_key in key and rm_key:
            print('remove key: %s' % key)
            continue
        if key_pre in key:
            out_key = key[key.find(rm_pre)+len(rm_pre):]
            out_state_dict[out_key] = value
            print('set key: %s --> out_key: %s' % (key, out_key))
    return out_state_dict


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def make_numpy_grid_singledim(tensor_data, padding=2, pad_value=0):
    tensor_data = tensor_data.detach()
    b, c, h, w = tensor_data.shape
    tensor_data = tensor_data.view([b*c, 1, h, w])
    vis = utils.make_grid(tensor_data, padding=padding, pad_value=pad_value)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    return vis[:, :, 0]


def make_numpy_grid_image_and_feature(tensor_images, tensor_features, padding=2,pad_value=0):
    tensor_images = tensor_images.detach().cpu()
    b1, c, h, w = tensor_images.shape
    assert c == 3
    tensor_feature = tensor_features.detach().cpu()
    b2,c,h,w = tensor_feature.shape
    assert b1 == b2
    tensor_feature = tensor_feature.view([b2*c,1,h,w])
    tensor_feature = torch.cat([tensor_feature,]*3,dim=1)
    tensor_data = torch.cat([tensor_images, tensor_feature],dim=0)
    vis = utils.make_grid(tensor_data, padding=padding, pad_value=pad_value)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(gpu_ids: str):
    # set gpu ids
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
    return gpu_ids


def write_dict_to_excel(out_path: str, contents: dict, sheet_name='Sheet1'):
    import xlwt
    """将字典contents写入excel文件中"""
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet(sheet_name)
    for i, item in enumerate(contents.items()):
        sheet1.write(0, i, item[0])
        sheet1.write(1, i, item[1])
    workbook.save(out_path)

import os
import functools


def get_item_form_txt(txt_file, item_name='precision_1'):
    """从txt文件中，读取出F1分数，以str形式返回"""
    out = []
    with open(txt_file, 'r') as file:
        for line in file.readlines():
            if item_name + ': ' in line:
                pos = line.find(item_name + ': ') + len(item_name) + 2
                value = line[pos:pos + 7]
                value = '%.3f' % (float(value)*100)
                out.append(value)
    single = True
    if single:
        return out[0]
    concat_values = functools.reduce(lambda a, b: a + '_' + b, out)
    return concat_values


def get_score_and_write_xls(base_folder, txt_name='log_eval.txt', xls_name='score.xls'):
    """从basefolder下的所有子目录中的对应txt文件中读取对应的值元素，
    并保存在对应文件夹下的xls中"""
    import glob
    all_paths = glob.glob(base_folder + '/**', recursive=True)
    val_dict = {}
    for path in all_paths:
        if os.path.isfile(path):
            if txt_name in path:
                print(f'process: {path}')
                base_root = os.path.dirname(path)
                txt_file = os.path.join(base_root, txt_name)
                out_xls_path = os.path.join(base_root, xls_name)
                val_dict['mf1'] = get_item_form_txt(txt_file, ' mf1')
                val_dict['precision_1'] = get_item_form_txt(txt_file, 'precision_1')
                val_dict['recall_1'] = get_item_form_txt(txt_file, 'recall_1')
                val_dict['F1_1'] = get_item_form_txt(txt_file, 'F1_1')
                val_dict['iou_1'] = get_item_form_txt(txt_file, 'iou_1')
                write_dict_to_excel(out_xls_path, val_dict)

def do_write_xls():
    base_root = r'G:\program\CD\CD4_3\checkpoints'
    get_score_and_write_xls(base_root)


if __name__ == '__main__':
    import os
    root = r'G:\program\CD\CD4_3\checkpoints\SSLM_simsiam2_fpn_m2_resnet18_sample16_syn1_imagenet_inria256_b64_lr0.01_pos0.1_train_pos0.1_val_400_poly_sgd'
    ckpt_path = os.path.join(root, 'best_ckpt_epoch_144.pt')
    out_path = os.path.join(root, 'pretrained_144.pth')
    convert_ckpt2pretrained2(ckpt_path, out_path)
    # convert_ckpt2pretrained2(ckpt_path, out_path, rm_pre='moco.encoder_q.')
    # do_write_xls()
