from argparse import ArgumentParser
import torch
import os

import utils
print(torch.cuda.is_available())
from misc.torchutils import seed_torch

"""
 train the CD model
"""


def do_downstream_task(**kwargs):
    from conf.config import do_cd_task
    cfg_path = kwargs['config_path']
    splits = kwargs.get('splits', ['train_5p', 'train_1p'])

    pretrained = kwargs.get('pretrained')
    head_pretrained = kwargs.get('head_pretrained')
    other_kwargs = {}
    other_kwargs['write2xls'] = kwargs.get('write2xls')
    other_kwargs['frozen_backbone_weights'] = kwargs.get('frozen_backbone_weights')
    other_kwargs['color_jet'] = kwargs.get('color_jet', 0)
    other_kwargs['scale_ratios'] = kwargs.get('scale_ratios', (1, 1))
    other_kwargs['scale_ratios_val'] = kwargs.get('scale_ratios', [1 if isinstance(s, list) else s for s in other_kwargs['scale_ratios']])
    other_kwargs['model_name'] = kwargs.get('model_name', None)
    other_kwargs['data_name'] = kwargs.get('data_name', None)
    other_kwargs['with_origin_scale_ret'] = kwargs.get('with_origin_scale_ret', 0)
    other_kwargs['with_sample'] = kwargs.get('with_sample', None)
    other_kwargs['scale_mode'] = kwargs.get('scale_mode', 0)
    other_kwargs['exchange_size'] = kwargs.get('exchange_size', 128)
    for split in splits:
        seed_torch(seed=2022)
        do_cd_task(cfg_path,
                   pretrained=pretrained,
                   head_pretrained=head_pretrained,
                   checkpoint_root=kwargs.get('checkpoint_root'),
                   split=split,
                   with_wandb=kwargs.get('with_wandb'),
                   project_task=kwargs.get('project_task'),
                   wandb_name=f"CD_{split}_{other_kwargs['model_name']}_{other_kwargs['scale_ratios']}",
                   subfolder_suffix=kwargs.get('subfolder_suffix', ''),
                   gpu_ids=kwargs.get('gpu_ids'),
                   **other_kwargs)


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='checkpoints/test', type=str)

    # logger
    parser.add_argument('--with_wandb',  type=int, default=0,
                        help="use wandb or not： 0： no use | >=1: use, 2: offline mode")
    parser.add_argument('--project_task',  type=str, default='cd',
                        help="which task is it?: cd? seg? ssl?, pretrain")

    parser.add_argument('--pretrained', default='imagenet', type=str,
                        help="imagenet | None | pretrain_path")
    parser.add_argument('--head_pretrained', default=False, type=bool,
                        help='head if pretrained? e.g., fcn | fpn head')
    parser.add_argument('--frozen_backbone_weights', default=False, action='store_true',
                        help='if frozen_backbone_weights? e.g., False | True | todo, layer names')

    parser.add_argument('--color_jet', default=0, type=bool,
                        help="if use color_jet augs.")

    parser.add_argument('--subfolder_suffix', default='', type=str,
                        help="")
    #  downstream task
    parser.add_argument('--config_path', default='./conf/cd_ours.yaml', type=str,
                        help="downstream task config file path: ./conf/cd_ours.yaml ")
    parser.add_argument('--data_name', default='LEVIR', type=str,
                        help="LEVIR  | SV_CD | DE_CD")
    parser.add_argument('--model_name', default=None, type=str,
                        help='base_fcn_resnet18 |'
                             'ifa_inter234_local4n_lpe_edgeconv_up2_resnet18_concat')
    parser.add_argument('--scale_ratios',  help='scale ratios for each temporal image',
                        default='1,0.25')
    parser.add_argument('--scale_mode', type=int, help='scale mode for downup scale augmentation, 0: whole, 1: crop area',
                        default=0)
    parser.add_argument('--exchange_size', type=int, help='size of bitemporal area exchange',
                        default=128)
    parser.add_argument('--splits', nargs='+', help='list of split: "train_5p, train_1p" ',
                        default=['train_1p', 'train_5p'])
    # Use like:
    # python main.py --splits train_5p, train_1p

    parser.add_argument('--write2xls', default=True, type=bool,
                        help='write scores into XX.xls')
    args = parser.parse_args()
    args.gpu_ids = utils.get_device(args.gpu_ids)
    args.scale_ratios = utils.str2int(args.scale_ratios)
    print(args.gpu_ids)
    print(args.frozen_backbone_weights)
    if args.with_wandb == 2:
        os.environ["WANDB_MODE"] = "dryrun"

    #  update pretrained path
    import data_config
    args.pretrained = data_config.get_pretrained_path(args.pretrained)
    print(args.scale_ratios, '.......................')
    # 增加下游任务
    seed_torch(seed=2022)
    do_downstream_task(**args.__dict__)


