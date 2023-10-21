import os
from typing import Dict, Sequence, Tuple, Optional, List, Union

from PIL import Image
import numpy as np

import torch
from torch.utils import data

from datasets.transforms import get_transforms, get_mask_transforms
from datasets.transforms import get_cd_augs

from misc.imutils import pil_downup_scale, random_crop_area_downup_scale

"""
some basic data loader
for example:
bitemporal image loader, change detection folder

data root
├─A
├─B
├─label
└─list
"""


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


class BiImageDataset(data.Dataset):
    """bitmeporal image dataset
    with_dataset_aug:
        when True, 在dataset中执行data augmentation，无需在后续trainer中对Tensor做aug；
            这时，basic transforms也为none，norm无效
        when False，在dataset中执行basic transforms转为tensor，后续在trainer中仍需要aug
    """
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 img_size: int =256,
                 norm: bool = False,
                 img_folder_names: Tuple[str, str] = ('A', 'B'),
                 list_folder_name: str = 'list',
                 scale_ratios: Tuple[Union[int, list], Union[int, list]] = (1, 1),
                 with_origin_scale_ret=0,
                 with_dataset_aug=False,
                 scale_mode=0,  # 0: normal downup；1：random crop area downup
                 exchange_size=128,
                 with_exchange=True,
                 ):
        super(BiImageDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split  # train | train_aug | val
        self.list_path = os.path.join(self.root_dir, list_folder_name, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)
        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.img_folder_names = img_folder_names
        self.img_size = img_size
        assert len(img_folder_names) == 2
        self.with_dataset_aug = with_dataset_aug
        if self.with_dataset_aug:
            self.basic_transforms = None
            from datasets.data_utils import CDDataAugmentation
            self.augs = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                # with_scale_random_crop=True,
                with_random_blur=True,
                # with_random_rot=True,
                # with_random_brightness=True,
                # with_random_gamma=True,
                # with_random_saturation=True,
                # with_random_contrast=True,
                # with_random_hue=True,
            )
        else:
            self.basic_transforms = get_transforms(norm=norm, img_size=img_size)
        self.scale_ratios = scale_ratios  # downsample/upsample ratios for each temporal image
        self.with_origin_scale_ret = with_origin_scale_ret
        assert self.with_origin_scale_ret in [0, 1, 2, 3]  # '0: no, 1: return A_ori, 2: return B_ori'
        # 3: return downscaled image with no upsampled
        if self.with_origin_scale_ret == 3:
            assert scale_mode == 0
        self.scale_mode = scale_mode
        self.exchange_size = exchange_size
        self.with_exchange = with_exchange
        # TODO: tmp
        if split is 'test' or split is 'val':
            self.with_exchange = False

    def _get_bi_images(self, name):
        imgs = dict()
        for i, (img_folder_name, scales) in enumerate(zip(['A', 'B'], self.scale_ratios)):
            A_path = os.path.join(self.root_dir, self.img_folder_names[i], name)
            img = np.asarray(Image.open(A_path).convert('RGB'))
            if self.with_origin_scale_ret == i+1:
                imgs[f'hr'] = img
            if isinstance(scales, list):
                if self.scale_mode == 4 or self.scale_mode == 5:  # 暂时用于表示patch augs
                    from misc.imutils import patch4_random_scale
                    img = patch4_random_scale(img, scales=scales, order=3)
                else:
                    # 对高分图做自适应降分扩增；
                    scale = scales[torch.randint(len(scales), (1,)).item()]
                    if scale != 1:
                        if self.scale_mode == 0:
                            img = pil_downup_scale(img, scale=scale, order=3)
                        elif self.scale_mode == 1 or self.scale_mode == 2:
                            crop_size = self.img_size // 2
                            img = random_crop_area_downup_scale(img, scale, cropsize=crop_size, order=3)
                        elif self.scale_mode == 3:
                            img = pil_downup_scale(img, scale=scale, order=3)
            else:
                scale = scales
                # 构造低分图
                if scale != 1:
                    if self.with_origin_scale_ret != 3:
                        img = pil_downup_scale(img, scale=scale, order=3)
                    else:
                        from misc.imutils import pil_rescale
                        img = pil_rescale(img, scale, order=3)
            # imgs.append(img)
            imgs[img_folder_name] = img

        if self.with_exchange:
        # if self.scale_mode == 2 or self.scale_mode == 3 or self.scale_mode == 5:  # 暂时用于表示 双时相crop区域交换
            from misc.imutils import random_crop_exchange
            # crop_size = self.img_size // 2
            crop_size = self.exchange_size
            if crop_size > 0:
                imgs['A'], imgs['B'] = random_crop_exchange(imgs['A'], imgs['B'], crop_size=crop_size)

        if self.basic_transforms is not None:
            # imgs = [self.basic_transforms(img) for img in imgs]
            imgs = {key: self.basic_transforms(img) for key, img in imgs.items()}
        return imgs

    def __getitem__(self, index):
        name = self.img_name_list[index % self.A_size]
        imgs_dict = self._get_bi_images(name)
        if self.with_dataset_aug:
            imgs, _ = self.augs.transform([img for img in imgs_dict.values()], [], to_tensor=True)
            imgs_dict = {key: img for img, key in zip(imgs, imgs_dict.keys())}
        imgs_dict.update({'name': name})
        return imgs_dict

        # return {'A': imgs[0],  'B': imgs[1], 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(BiImageDataset):
    '''
    注意：这里仅应用基础的transforms，即tensor化，resize等
        其他transforms在外部的augs中应用
    '''
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 img_size: int = 256,
                 norm: bool = False,
                 img_folder_names: Tuple[str, str] = ('A', 'B'),
                 list_folder_name: str = 'list',
                 label_transform: str = 'norm',
                 label_folder_name: str = 'label',
                 scale_ratios: Tuple[Union[int, list], Union[int, list]] = (1, 1),
                 with_origin_scale_ret=0,
                 with_dataset_aug=False,  # 在dataset里面加aug，或在trainer中对tensor加aug
                 scale_mode=0,
                 exchange_size=128,
                 ):
        super(CDDataset, self).__init__(root_dir, split=split,
                                        img_folder_names=img_folder_names,
                                        list_folder_name=list_folder_name,
                                        img_size=img_size,
                                        norm=norm,
                                        scale_ratios=scale_ratios,
                                        with_origin_scale_ret=with_origin_scale_ret,
                                        with_dataset_aug=with_dataset_aug,
                                        scale_mode=scale_mode,
                                        exchange_size=exchange_size)
        self.label_folder_name = label_folder_name
        self.label_transform = label_transform
        self.with_dataset_aug = with_dataset_aug
        if self.with_dataset_aug is False:
            self.basic_mask_transforms = get_mask_transforms(img_size=img_size)
        else:
            self.basic_mask_transforms = None

    def _get_label(self, name):
        mask_path = os.path.join(self.root_dir, self.label_folder_name, name)
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            mask = mask // 255
        elif self.label_transform == 'ignore0_sub1':
            mask = mask - 1
            # 原来label==0的部分变为255，自动被ignore
        if self.basic_mask_transforms is not None:
            mask = self.basic_mask_transforms(mask)
        return mask

    def __getitem__(self, index):
        name = self.img_name_list[index]
        imgs_dict = self._get_bi_images(name)
        mask = self._get_label(name)
        if self.with_dataset_aug:
            imgs, [mask] = self.augs.transform([img for img in imgs_dict.values()], [mask], to_tensor=True)
            imgs_dict = {key: img for img, key in zip(imgs, imgs_dict.keys())}
        imgs_dict.update({'mask': mask, 'name': name})
        return imgs_dict

    def get_item_by_name(self, name):
        imgs_dict = self._get_bi_images(name)
        mask = self._get_label(name)
        if self.with_dataset_aug:
            imgs, [mask] = self.augs.transform([img for img in imgs_dict.values()], [mask], to_tensor=True)
            imgs_dict = {key: img for img, key in zip(imgs, imgs_dict.keys())}
        imgs_dict.update({'mask': mask, 'name': name})
        return imgs_dict


class SCDDataset(BiImageDataset):
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 img_size: int = 256,
                 norm: bool = False,
                 img_folder_names: Tuple[str, str] = ('A', 'B'),
                 list_folder_name: str = 'list',
                 label_folder_names: Tuple[str, ] = ('label1_gray', 'label2_gray'),
                 label_transform: str = 'norm',
                 scale_ratios: Tuple[Union[int, list], Union[int, list]] = (1, 1),
                 with_origin_scale_ret=0):
        super(SCDDataset, self).__init__(root_dir, split=split,
                                        img_folder_names=img_folder_names,
                                        list_folder_name=list_folder_name,
                                        img_size=img_size,
                                        norm=norm,scale_ratios=scale_ratios,
                                        with_origin_scale_ret=with_origin_scale_ret)
        self.basic_mask_transforms = get_mask_transforms(img_size=img_size)
        self.label_folder_names = label_folder_names
        self.label_transform = label_transform

    def _get_labels(self, name):
        masks = []
        for label_folder_name in self.label_folder_names:
            mask_path = os.path.join(self.root_dir, label_folder_name, name)
            mask = np.array(Image.open(mask_path), dtype=np.uint8)
            masks.append(mask)
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            masks = [mask // 255 for mask in masks]
        if self.basic_mask_transforms is not None:
            masks = [self.basic_mask_transforms(mask) for mask in masks]
        return masks

    def __getitem__(self, index):
        name = self.img_name_list[index]
        imgs_dict = self._get_bi_images(name)
        masks = self._get_labels(name)
        imgs_dict.update({'mask1': masks[0], 'mask2': masks[1], 'name': name})
        return imgs_dict


from misc.torchutils import visualize_tensors


def get_image_by_name(dataset: CDDataset, name):
    item_dict = dataset.get_item_by_name(name)
    A, B, M = item_dict['A'], item_dict['B'], item_dict['mask']
    item_dict['A'] = A.unsqueeze(dim=0)
    item_dict['B'] = B.unsqueeze(dim=0)
    item_dict['mask'] = M.unsqueeze(dim=0) * 255

    # visualize_tensors(A, B, M)
    from misc.torchutils import save_visuals
    out_folder = r'../vis/tmp'
    os.makedirs(out_folder, exist_ok=True)
    save_visuals(item_dict, img_dir=out_folder, name=item_dict['name'], if_normalize=True)


def iter_dataset(dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(0),
        drop_last=True)
    augs = get_cd_augs(imgz_size=256)

    for i, batch in enumerate(dataloader):
        A0 = batch['A']
        B0 = batch['B']
        M0 = batch['mask']
        # A, B, L = augs(batch['A'], batch['B'], batch['mask'])
        # A = batch['A']
        # L = batch['mask']
        # print(A.max())
        # print(L.max())
        # A2 = batch['A2']
        # L2 = batch['L2']
        # mask = batch['seg_mask']
        # visualize_tensors(A0[0], B0[0], A[0], B[0], L[0])
        visualize_tensors(A0[0], B0[0], M0[0])


if __name__ == '__main__':
    is_train = True
    root_dir = r'G:/tmp_data/inria_cut256/'
    root_dir = 'D:/dataset/CD/LEVIR-CD/cut/'

    split = 'test'
    label_transform = 'norm'
    from misc.torchutils import seed_torch
    seed_torch(2023)
    # dataset = BiImageDataset(root_dir=root_dir, split=split,)
    dataset = CDDataset(root_dir=root_dir, split=split,
                         label_transform=label_transform,
                        scale_ratios=[[1, 0.5, 0.25], 0.25],
                        with_dataset_aug=False, scale_mode=3,
                        norm=True)
    name = 'test_11_0512_0512.png'
    # get_image_by_name(dataset, name)
    iter_dataset(dataset)