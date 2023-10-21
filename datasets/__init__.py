import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from typing import Optional

import data_config
from datasets.cd_dataset import BiImageDataset, CDDataset, SCDDataset
from datasets.base_dataset import ImageDataset, SegDataset


def worker_init_fn(worker_id):
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(is_train=False,
               batch_size: int = 8,
               num_workers: int = 4,
               img_size: int = 256,
               dataset_type: str = 'CDDataset',
               split: str = 'test',
               data_name: str = 'LEVIR',
               norm: Optional[bool] = None,
               scale_ratios=(1, 1),
               with_dataset_aug=False,
               scale_mode=0,
               exchange_size=128,
               **kwargs):
    #  添加norm选项，修复validation的时候，train（做了augs）与val（没做augs）数据不一致的情况。
    if norm is None:
        norm = not is_train
        # norm = False
        print(f'Norm: {norm}')
    data_set = get_dataset(data_name, dataset_type, split, img_size, norm=norm,
                           scale_ratios=scale_ratios, with_dataset_aug=with_dataset_aug,
                           scale_mode=scale_mode, exchange_size=exchange_size)
    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            )
    # persistent_workers
    return dataloader


def get_loaders(batch_size: int = 8,
                num_workers: int = 4,
                img_size: int = 256,
                dataset_type: str = 'CDDataset',
                split: str = 'train',
                data_name: str = 'LEVIR',
                **kwargs):
    split_val = kwargs.get('split_val', 'val')
    data_name_val = kwargs.get('data_name_val', data_name)
    img_size_val = kwargs.get('img_size_val', img_size)
    persistent_workers = kwargs.get('persistent_workers', True)
    # TODO
    # 20220331: ssl任务，val_set也不需要norm；而cd任务val_set需要norm
    val_data_norm = kwargs.get('val_data_norm', True)
    scale_ratios = kwargs.get('scale_ratios', (1, 1))
    if isinstance(scale_ratios, tuple) or isinstance(scale_ratios, list):
        scale_ratios_val = kwargs.get('scale_ratios_val', [1 if isinstance(s, list) else s for s in scale_ratios])
    else:
        scale_ratios_val = kwargs.get('scale_ratios_val', (1, 1))
    with_origin_scale_ret = kwargs.get('with_origin_scale_ret', 0)
    with_dataset_aug = kwargs.get('with_dataset_aug', False)
    scale_mode = kwargs.get('scale_mode', 0)
    training_set = get_dataset(data_name, dataset_type, split, img_size=img_size,
                               scale_ratios=scale_ratios, scale_mode=scale_mode,
                               with_origin_scale_ret=with_origin_scale_ret, with_dataset_aug=with_dataset_aug)
    val_set = get_dataset(data_name_val, dataset_type, split_val,
                          img_size=img_size_val, norm=val_data_norm,
                          scale_ratios=scale_ratios_val)
    datasets = {'train': training_set, 'val': val_set}
    g = torch.Generator()
    g.manual_seed(0)
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size,
                                 shuffle=(x == 'train'),
                                 num_workers=num_workers, pin_memory=True,
                                 drop_last=True,
                                 persistent_workers=persistent_workers,
                                 worker_init_fn=worker_init_fn,
                                 generator=g)
                   for x in ['train', 'val']}
    return dataloaders


def get_one_dataset(data_name, dataset_type, split, img_size, norm=False,
                    scale_ratios=(1, 1), with_origin_scale_ret=0, with_dataset_aug=False,
                    scale_mode=0, exchange_size=128):
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    label_folder_name = dataConfig.label_folder_name
    if dataset_type == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, split=split, img_size=img_size,
                             label_folder_name=label_folder_name,
                             label_transform=label_transform, norm=norm,
                             scale_ratios=scale_ratios,
                             with_origin_scale_ret=with_origin_scale_ret,
                             with_dataset_aug=with_dataset_aug,
                             scale_mode=scale_mode, exchange_size=exchange_size)
    elif dataset_type == 'SegDataset':
        data_set = SegDataset(root_dir=root_dir, split=split, img_size=img_size,
                              label_folder_name=label_folder_name,
                              label_transform=label_transform, norm=norm,
                              scale_ratios=scale_ratios)
    elif dataset_type == 'BiImageDataset':
        data_set = BiImageDataset(root_dir=root_dir, split=split, img_size=img_size
                                , norm=norm, scale_ratios=scale_ratios,
                                  with_origin_scale_ret=with_origin_scale_ret,
                                  with_dataset_aug=with_dataset_aug,
                                  scale_mode=scale_mode, exchange_size=exchange_size)
    elif dataset_type == 'ImageDataset':
        data_set = ImageDataset(root_dir=root_dir, split=split, img_size=img_size,
                                norm=norm, scale_ratios=scale_ratios)
    elif dataset_type == 'SCDDataset':
        label_folder_names = dataConfig.label_folder_names
        data_set = SCDDataset(root_dir=root_dir, split=split, img_size=img_size,
                             label_folder_names=label_folder_names,
                             label_transform=label_transform, norm=norm,
                              scale_ratios=scale_ratios,
                              with_origin_scale_ret=with_origin_scale_ret)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % dataset_type)
    return data_set


def get_dataset(data_name,  dataset_type, split, img_size, norm=False, scale_ratios=(1, 1),
                with_origin_scale_ret=0, with_dataset_aug=False, scale_mode=0,
                exchange_size=128, with_exchange=True):
    if len(data_name.split(',')) > 1:
        dataset_list = []
        for data_name_ in data_name.split(','):
            #  非空
            if data_name_:
                dataset_list.append(
                    get_one_dataset(data_name_, dataset_type, split, img_size, norm,
                                    scale_ratios, with_origin_scale_ret, with_dataset_aug,
                                    scale_mode, exchange_size)
                )
        return ConcatDataset(dataset_list)
    else:
        return get_one_dataset(data_name,  dataset_type, split, img_size,
                               norm, scale_ratios, with_origin_scale_ret,
                               with_dataset_aug, scale_mode, exchange_size)


if __name__ == '__main__':
    pass
