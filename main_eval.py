from argparse import ArgumentParser
import torch
import os

import utils
print(torch.cuda.is_available())


"""
CD model Validation
support validating bitemporal images with varying resolution difference ratios (scale_ratios)
"""


def do_downstream_task(**kwargs):
    from conf.config import do_cd_eval
    # cfg_path = './conf/cd_ours.yaml'
    cfg_path = kwargs['config_path']
    splits = kwargs.get('splits', ['test'])
    other_kwargs = {}
    # other_kwargs['scale_ratios'] = kwargs.get('scale_ratios', [(1, 1)])
    other_kwargs['model_name'] = kwargs.get('model_name', None)
    other_kwargs['data_name'] = kwargs.get('data_name', None)
    for split in splits:
        for scale_ratios in kwargs.get('scale_ratios'):
            other_kwargs['scale_ratios'] = scale_ratios
            other_kwargs['write2xls'] = kwargs.get('write2xls') + f"_s{scale_ratios}"
            do_cd_eval(cfg_path,
                       checkpoint_dir=kwargs.get('checkpoint_dir'),
                       split=split,
                       gpu_ids=kwargs.get('gpu_ids'),
                       **other_kwargs)


def do_for_one_folder(args):
    # TODO：you can change the scales_ratios as you want
    if args.data_name == 'LEVIR':
        args.scale_ratios = [(1, 1), (1, 0.75), (1, 0.5),  (1, 1 / 3), (1, 0.25),  (1, 0.2), (1, 1 / 6), (1, 0.125)]
    elif args.data_name == 'SV_CD':
        args.scale_ratios = [(1, 1), (1, 0.75), (1, 0.5), (1, 0.25), (1, 0.2), (1, 0.125), (1, 1 / 9), (1, 0.1),
                             (1, 1 / 12)]
    elif args.data_name == 'DE_CD':
        args.scale_ratios = [(1, 1), (1, 3 / 10 * 3), (1, 3 / 5), (1, 4 / 10), (1, 3 / 10), (0.25, 1), (1 / 5, 1),
                             (1 / 6, 1)]
    print(args.scale_ratios, '.......................')
    try:
        do_downstream_task(**args.__dict__)
    except Exception as e:
        print(e)




if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='checkpoints/test', type=str)

    # logger

    #  downstream task
    parser.add_argument('--config_path', default='./conf/cd_ours.yaml', type=str,
                        help="downstream task config file path ")
    parser.add_argument('--model_name', default='ifa_inter234_local4n_lpe_edgeconv_up2_resnet18_concat', type=str,
                        help='base_fcn_resnet18 | ifa_inter234_local4n_lpe_edgeconv_up2_resnet18_concat')
    parser.add_argument('--data_name', default='LEVIR', type=str,)
    parser.add_argument('--checkpoint_dir', default='checkpoints/ours_levir1x', type=str, help='model checkpoint to be load for eval')
    parser.add_argument('--scale_ratios',  help='scale ratios for each temporal image',
                        default=None )
    parser.add_argument('--splits', nargs='+', help='list of split: "train_5p, train_1p" ',
                        default=['test'])
    # Use like:
    parser.add_argument('--write2xls', default='test_scores', type=str,
                        help='write scores into XX.xls')
    args = parser.parse_args()
    args.gpu_ids = utils.get_device(args.gpu_ids)
    # args.scale_ratios = utils.str2int(args.scale_ratios)
    print(args.gpu_ids)
    args.pretrained = 'none'
    if os.path.exists(os.path.join(args.checkpoint_dir, 'best_ckpt.pt')):
        do_for_one_folder(args)
    else:
        raise FileNotFoundError(os.path.join(args.checkpoint_dir, 'best_ckpt.pt'))