import os
from typing import Dict, Sequence, Tuple, Optional, Union

from PIL import Image
import numpy as np

import torch
from torch.utils import data

from datasets.transforms import get_transforms, get_mask_transforms
from datasets.transforms import get_seg_augs

"""
some basic data loader
for example:
Image loader, Segmentation loader, 

data root
├─A
├─label
└─list
"""


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


class ImageDataset(data.Dataset):
    """list dataloder"""
    def __init__(self, root_dir: str,
                 split: str = 'train',
                 img_size: int = 256,
                 norm: bool = False,
                 img_folder_name: Union[str, list, tuple] = 'A',
                 list_folder_name: str = 'list',
                 scale_ratios: Union[int, list] = 1):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split  # train | train_aug | val
        self.list_path = os.path.join(self.root_dir, list_folder_name, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)
        if isinstance(img_folder_name, list) or isinstance(img_folder_name, tuple):
            # 此处为了兼容存在多个img_folder，内部文件共名字的情况，比如img_folder_name=['A','B']
            self.img_folder_with_name_list = [img_folder_name_+'/'+name
                                     for name in self.img_name_list
                                     for img_folder_name_ in img_folder_name]
        elif isinstance(img_folder_name, str):
            self.img_folder_with_name_list = [img_folder_name+'/'+name
                                     for name in self.img_name_list]
        else:
            raise NotImplementedError
        self.A_size = len(self.img_folder_with_name_list)  # get the size of dataset A
        self.img_folder_name = img_folder_name
        self.img_size = img_size
        self.norm = norm
        self.basic_transforms = get_transforms(norm=norm, img_size=img_size)
        self.scale_ratios = scale_ratios

    def __getitem__(self, index):
        folder_with_name = self.img_folder_with_name_list[index % self.A_size]
        img_folder_name = folder_with_name.split('/')[0]
        name = folder_with_name.split('/')[-1]
        A_path = os.path.join(self.root_dir, img_folder_name, name)
        img = np.asarray(Image.open(A_path).convert('RGB'))
        scales = self.scale_ratios
        if isinstance(scales, list):
            scale = scales[torch.randint(len(scales), (1,)).item()]
        else:
            scale = scales
        if scale != 1:
            from misc.imutils import pil_rescale, pil_resize
            h, w = img.shape[:2]
            img = pil_rescale(img, scale=scale, order=3)
            img = pil_resize(img, size=[h, w], order=3)
        if self.basic_transforms is not None:
            img = self.basic_transforms(img)

        return {'A': img,  'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class SegDataset(ImageDataset):
    '''
    transforms: 表示同时对image 和 mask 做变换；
    '''
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 img_size: int = 256,
                 norm: bool = False,
                 img_folder_name: Union[str, list, tuple] = 'A',
                 label_transform: str = 'norm',
                 label_folder_name: str = 'label',
                 scale_ratios: Union[int, list] = 1):
        super(SegDataset, self).__init__(root_dir, split=split,
                                         img_size=img_size,
                                         norm=norm,
                                         img_folder_name=img_folder_name,
                                         scale_ratios=scale_ratios)
        self.basic_mask_transforms = get_mask_transforms(img_size=img_size)
        self.label_folder_name = label_folder_name
        self.label_transform = label_transform

    def __getitem__(self, index):
        # name = self.img_name_list[index]
        # A_path = os.path.join(self.root_dir, self.img_folder_name, name)
        folder_with_name = self.img_folder_with_name_list[index % self.A_size]
        img_folder_name = folder_with_name.split('/')[0]
        name = folder_with_name.split('/')[-1]
        A_path = os.path.join(self.root_dir, img_folder_name, name)

        img = np.asarray(Image.open(A_path).convert('RGB'))
        scales = self.scale_ratios
        if isinstance(scales, list):
            scale = scales[torch.randint(len(scales), (1,)).item()]
        else:
            scale = scales
        if scale != 1:
            from misc.imutils import pil_rescale, pil_resize
            h, w = img.shape[:2]
            img = pil_rescale(img, scale=scale, order=3)
            img = pil_resize(img, size=[h, w], order=3)

        L_path = os.path.join(self.root_dir, self.label_folder_name, name)
        mask = np.array(Image.open(L_path), dtype=np.uint8)

        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            mask = mask // 255
        elif self.label_transform == 'ignore0_sub1':
            mask = mask - 1
            # 原来label==0的部分变为255，自动被ignore

        if self.basic_transforms is not None:
            img = self.basic_transforms(img)
        if self.basic_mask_transforms is not None:
            mask = self.basic_mask_transforms(mask)

        return {'A': img, 'mask': mask, 'name': name}


from misc.torchutils import visualize_tensors


if __name__ == '__main__':
    is_train = True
    root_dir = r'G:/tmp_data/inria_cut256/'
    img_folder_name = ['A']
    split = 'train'
    label_transform = 'norm'
    dataset = SegDataset(root_dir=root_dir, split=split,
                         img_folder_name=img_folder_name,
                         label_transform=label_transform)
    print(f'dataset len is {len(dataset)}')
    augs = get_seg_augs(imgz_size=256)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(0),
        drop_last=True)

    for i, batch in enumerate(dataloader):
        A0 = batch['A']
        L0 = batch['mask']
        A, L = augs(batch['A'], batch['mask'])
        # A = batch['A']
        # L = batch['mask']
        print(A.max())
        print(L.max())
        print(L0.max())
        # A2 = batch['A2']
        # L2 = batch['L2']
        # mask = batch['seg_mask']
        visualize_tensors(A0[0], A[0], L0[0], L[0])
