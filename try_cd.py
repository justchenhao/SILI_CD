from argparse import ArgumentParser
import torch
import os

import utils
print(torch.cuda.is_available())


"""
CD model test 
support testing different scale_ratios
with the prediction results saved in the local folder.
"""


def do_downstream_task(**kwargs):
    from conf.config import do_cd_eval
    cfg_path = kwargs['config_path']
    splits = kwargs.get('splits', ['try'])
    other_kwargs = {}
    other_kwargs['model_name'] = kwargs.get('model_name', None)
    other_kwargs['data_name'] = kwargs.get('data_name', None)
    other_kwargs['eval_samples_num'] = kwargs.get('eval_samples_num', 10e8)
    for split in splits:
        for scale_ratios in kwargs.get('scale_ratios'):
            other_kwargs['scale_ratios'] = scale_ratios
            other_kwargs['write2xls'] = False
            other_kwargs['pred_dir'] = kwargs.get('checkpoint_dir') + \
                                       f'/pred_{scale_ratios}'.replace(' ', '_')
            do_cd_eval(cfg_path,
                       checkpoint_dir=kwargs.get('checkpoint_dir'),
                       split=split,
                       gpu_ids=kwargs.get('gpu_ids'),
                       **other_kwargs)


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # logger
    #  downstream task
    parser.add_argument('--config_path', default='./conf/cd_ours.yaml', type=str,
                        help="downstream task config file path ")
    parser.add_argument('--model_name', default='ifa_inter234_local4n_lpe_edgeconv_up2_resnet18_concat', type=str,
                        help='base_fcn_resnet18 | ifa_resnet18_concat')
    parser.add_argument('--data_name', default='levir_try', type=str,)
    parser.add_argument('--eval_samples_num', default=500, type=int, help='eval samples num')
    parser.add_argument('--checkpoint_dir', default='checkpoints/ours_levir1x', type=str, help='model checkpoint to be load for eval')
    parser.add_argument('--scale_ratios',  help='scale ratios for each temporal image',
                        default=None
                        )
    parser.add_argument('--splits', nargs='+', help='list of split: "train_5p, train_1p" ',
                        default=['try'])
    # Use like:
    parser.add_argument('--write2xls', default='test_scores', type=str,
                        help='write scores into XX.xls')
    args = parser.parse_args()
    args.gpu_ids = utils.get_device(args.gpu_ids)
    # args.scale_ratios = utils.str2int(args.scale_ratios)
    print(args.gpu_ids)

    args.pretrained = 'none'
    args.scale_ratios = [(1, 1), (1, 0.75), (1, 0.5), (1, 1 / 3), (1, 0.25), (1, 0.2), (1, 1 / 6), (1, 0.125)]

    do_downstream_task(**args.__dict__)