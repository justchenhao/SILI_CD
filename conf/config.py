from omegaconf import DictConfig, OmegaConf
import os
from typing import Union, Optional


def do_cd_eval(
        default_config_path: str = './cd_ours.yaml',
        checkpoint_dir: str = '',
        split: Optional[str] = 'test',
        gpu_ids: list = [0],
        **other_kwargs,):
    from datasets import get_loader
    from models.evaluator import CDEvaluator
    from omegaconf import DictConfig, OmegaConf
    cfg = OmegaConf.load(default_config_path)
    kwargs = dict(cfg)
    scale_ratios = other_kwargs.get('scale_ratios', (1, 1))
    kwargs['checkpoint_dir'] = checkpoint_dir
    #  参数调整
    if other_kwargs.get('model_name') is not None and other_kwargs.get('model_name') is not 'None':
        kwargs['model_name'] = other_kwargs['model_name']
    if other_kwargs.get('data_name') is not None and other_kwargs.get('data_name') is not 'None':
        kwargs['data_name'] = other_kwargs['data_name']
    kwargs['gpu_ids'] = gpu_ids
    kwargs['write2xls'] = other_kwargs.get('write2xls', False)
    kwargs['pred_dir'] = other_kwargs.get('pred_dir', None)
    kwargs['eval_samples_num'] = other_kwargs.get('eval_samples_num', 10e8)
    # test
    dataloader = get_loader(data_name=kwargs['data_name'],
                            img_size=kwargs['img_size'],
                            batch_size=kwargs['batch_size'],
                            is_train=False,
                            split=split,
                            scale_ratios=scale_ratios,
                            with_dataset_aug=False)
    kwargs['dataloader'] = dataloader
    model = CDEvaluator(**kwargs)
    model.eval_models()


def do_cd_task(
        default_config_path: str = './cd_ours.yaml',
        pretrained: Union[str, bool] = 'imagenet',
        head_pretrained: bool = False,
        checkpoint_root: str = 'checkpoints',
        split: Optional[str] = None,
        with_wandb: str = False,
        project_task: str = 'test',
        wandb_name: str = 'test',
        subfolder_suffix: str = '',
        gpu_ids: list = [0],
        **other_kwargs,
):
    """
    @param default_config_path: cd的默认配置文件*.yaml
    @param pretrained: 预训练模型路径 OR imagenet等str
    @param checkpoint_root: 保存的根路径
    @param split: 训练集split：train | train_5
    @return:
    """
    from models.trainer import CDTrainer, CDTrainer_SR
    from models.evaluator import CDEvaluator
    from omegaconf import DictConfig, OmegaConf
    from datasets import get_loader, get_loaders
    from misc.draw_tool import save_acc_curve
    # cfg_path = './conf/cd_ours.yaml'
    # import yaml
    # f = open(default_config_path, 'r+')
    # kwargs = yaml.load(f, Loader=yaml.FullLoader)
    cfg = OmegaConf.load(default_config_path)
    kwargs = dict(cfg)
    scale_ratios = other_kwargs.get('scale_ratios', (1, 1))
    scale_ratios_val = other_kwargs.get('scale_ratios_val', [1 if isinstance(s, list) else s for s in scale_ratios])

    if split is not None:
        kwargs['split'] = split
    kwargs['checkpoint_root'] = checkpoint_root
    #  参数调整
    if other_kwargs.get('model_name') is not None and other_kwargs.get('model_name') != 'None':
        kwargs['model_name'] = other_kwargs['model_name']
    else:
        other_kwargs['model_name'] = kwargs['model_name']
    if other_kwargs.get('data_name') is not None and other_kwargs.get('data_name') != 'None':
        kwargs['data_name'] = other_kwargs['data_name']
    else:
        other_kwargs['data_name'] = kwargs['data_name']
    kwargs['pretrained'] = pretrained
    if len(str(scale_ratios).split(',')) > 6:
        import hashlib
        def create_id(data):
            m = hashlib.md5(str(data).encode("utf-8"))
            return m.hexdigest()
        scale_ratios = create_id(scale_ratios)
    kwargs['project_name'] = 'CD_' + kwargs['model_name'] + '_' \
                             + kwargs['data_name'] + '_'+str(scale_ratios) + '_'\
                             + 'b'+str(kwargs['batch_size']) + '_'\
                             + str(kwargs['lr']) + '_'\
                             + kwargs['split'] + '_'\
                             + kwargs['split_val'] + '_'\
                             + kwargs['optim_mode'] + '_'\
                             + kwargs['lr_policy'] + '_'\
                             + str(kwargs['max_epochs'])
    kwargs['head_pretrained'] = head_pretrained
    if head_pretrained:
        kwargs['project_name'] = kwargs['project_name'] + '_headpre'
    # print(other_kwargs['frozen_backbone_weights'],"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if 'frozen_backbone_weights' in other_kwargs and other_kwargs['frozen_backbone_weights'] == True:
        kwargs['project_name'] = kwargs['project_name'] + '_frozen'
    if 'color_jet' in other_kwargs and other_kwargs['color_jet'] != 0:
        kwargs['project_name'] = kwargs['project_name'] + '_cj'
    if kwargs.get('with_dataset_aug', False):
        kwargs['project_name'] = kwargs['project_name'] + '_da1'  #da: has scale aug, da1: no scale
    if other_kwargs.get('scale_mode', 0) != 0:
        kwargs['project_name'] = kwargs['project_name'] + f"_cs{other_kwargs.get('scale_mode', 0)}"
    if other_kwargs.get('exchange_size', 128) != 128:
        kwargs['project_name'] = kwargs['project_name'] + f"_exs{other_kwargs.get('exchange_size', 128)}"

    kwargs['project_name'] += subfolder_suffix
    kwargs['checkpoint_dir'] = os.path.join(checkpoint_root, kwargs['project_name'])
    os.makedirs(kwargs['checkpoint_dir'], exist_ok=True)
    #  visualize dir
    kwargs['vis_dir'] = os.path.join('vis', kwargs['project_name'])
    os.makedirs(kwargs['vis_dir'], exist_ok=True)

    kwargs.update(**other_kwargs)
    #  train
    dataloaders = get_loaders(**kwargs)
    kwargs['dataloaders'] = dataloaders
    kwargs['with_wandb'] = with_wandb
    kwargs['project_task'] = project_task
    kwargs['wandb_name'] = wandb_name
    kwargs['gpu_ids'] = gpu_ids
    if other_kwargs.get('with_origin_scale_ret', 0) == 1 or other_kwargs.get('with_origin_scale_ret', 0) == 2 or other_kwargs.get('with_sample', None) == 1:
        model = CDTrainer_SR(**kwargs)
    else:
        model = CDTrainer(**kwargs)
    model.train_models()

    # test
    dataloader = get_loader(data_name=kwargs['data_name'],
                            img_size=kwargs['img_size'],
                            batch_size=kwargs['batch_size'],
                            is_train=False,
                            split='test',
                            scale_ratios=scale_ratios_val,
                            with_dataset_aug=False,
                            scale_mode=0)
    kwargs['dataloader'] = dataloader
    model = CDEvaluator(**kwargs)
    model.eval_models()
    # draw
    save_acc_curve(project_name=kwargs['project_name'],
                   checkpoint_root=kwargs['checkpoint_root'])


def do_cd_train(
        default_config_path: str = './cd_ours.yaml',
        pretrained: Union[str, bool] = 'imagenet',
        head_pretrained: bool = False,
        checkpoint_root: str = 'checkpoints',
        split: Optional[str] = None,
        with_wandb: str = False,
        project_task: str = 'test',
        wandb_name: str = 'test',
        subfolder_suffix: str = '',
        gpu_ids: list = [0],
        **other_kwargs,
):
    """
    @param default_config_path: cd的默认配置文件*.yaml
    @param pretrained: 预训练模型路径 OR imagenet等str
    @param checkpoint_root: 保存的根路径
    @param split: 训练集split：train | train_5
    @return:
    """
    from models.trainer import CDTrainer, CDTrainer_SR
    from omegaconf import DictConfig, OmegaConf
    from datasets import get_loader, get_loaders

    cfg = OmegaConf.load(default_config_path)
    kwargs = dict(cfg)
    scale_ratios = other_kwargs.get('scale_ratios', (1, 1))
    scale_ratios_val = other_kwargs.get('scale_ratios_val', [1 if isinstance(s, list) else s for s in scale_ratios])
    kwargs['max_epochs'] = 1  #TODO
    if split is not None:
        kwargs['split'] = split
    kwargs['checkpoint_root'] = checkpoint_root
    #  参数调整
    if other_kwargs.get('model_name') is not None and other_kwargs.get('model_name') != 'None':
        kwargs['model_name'] = other_kwargs['model_name']
    else:
        other_kwargs['model_name'] = kwargs['model_name']
    if other_kwargs.get('data_name') is not None and other_kwargs.get('data_name') != 'None':
        kwargs['data_name'] = other_kwargs['data_name']
    else:
        other_kwargs['data_name'] = kwargs['data_name']
    kwargs['pretrained'] = pretrained
    if len(str(scale_ratios).split(',')) > 6:
        import hashlib
        def create_id(data):
            m = hashlib.md5(str(data).encode("utf-8"))
            return m.hexdigest()
        scale_ratios = create_id(scale_ratios)
    kwargs['project_name'] = 'CD_' + kwargs['model_name'] + '_' \
                             + kwargs['data_name'] + '_'+str(scale_ratios) + '_'\
                             + 'b'+str(kwargs['batch_size']) + '_'\
                             + str(kwargs['lr']) + '_'\
                             + kwargs['split'] + '_'\
                             + kwargs['split_val'] + '_'\
                             + kwargs['optim_mode'] + '_'\
                             + kwargs['lr_policy'] + '_'\
                             + str(kwargs['max_epochs'])
    kwargs['head_pretrained'] = head_pretrained
    if head_pretrained:
        kwargs['project_name'] = kwargs['project_name'] + '_headpre'
    # print(other_kwargs['frozen_backbone_weights'],"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if 'frozen_backbone_weights' in other_kwargs and other_kwargs['frozen_backbone_weights'] == True:
        kwargs['project_name'] = kwargs['project_name'] + '_frozen'
    if 'color_jet' in other_kwargs and other_kwargs['color_jet'] != 0:
        kwargs['project_name'] = kwargs['project_name'] + '_cj'
    if kwargs.get('with_dataset_aug', False):
        kwargs['project_name'] = kwargs['project_name'] + '_da1'  #da: has scale aug, da1: no scale
    if other_kwargs.get('scale_mode', 0) != 0:
        kwargs['project_name'] = kwargs['project_name'] + f"_cs{other_kwargs.get('scale_mode', 0)}"
    if other_kwargs.get('exchange_size', 128) != 128:
        kwargs['project_name'] = kwargs['project_name'] + f"_exs{other_kwargs.get('exchange_size', 128)}"

    kwargs['project_name'] += subfolder_suffix
    kwargs['checkpoint_dir'] = os.path.join(checkpoint_root, kwargs['project_name'])
    os.makedirs(kwargs['checkpoint_dir'], exist_ok=True)
    #  visualize dir
    kwargs['vis_dir'] = os.path.join('vis', kwargs['project_name'])
    os.makedirs(kwargs['vis_dir'], exist_ok=True)

    kwargs.update(**other_kwargs)
    #  train
    dataloaders = get_loaders(**kwargs)
    kwargs['dataloaders'] = dataloaders
    kwargs['with_wandb'] = with_wandb
    kwargs['project_task'] = project_task
    kwargs['wandb_name'] = wandb_name
    kwargs['gpu_ids'] = gpu_ids
    if other_kwargs.get('with_origin_scale_ret', 0) == 1 or other_kwargs.get('with_origin_scale_ret', 0) == 2 or other_kwargs.get('with_sample', None) == 1:
        model = CDTrainer_SR(**kwargs)
    else:
        model = CDTrainer(**kwargs)
    model.train_models(do_eval=False)


if __name__ == '__main__':
    cfg_path = 'cd_ours.yaml'
    cfg = OmegaConf.load(cfg_path)
    print(cfg)
    do_cd_task(cfg_path)
