import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

import kornia.augmentation as K

##########################################################
# basic_transform: toTensor, /norm, Resize
##########################################################


def get_transforms(norm=False, img_size=256):
    basic_transform = []
    basic_transform.append(T.ToTensor())  # ndarray转为 torch.FloatTensor， 范围[0,1]
    if norm:
        basic_transform.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    basic_transform.append(T.Resize(size=(img_size, img_size), interpolation=InterpolationMode.BILINEAR))
    return T.Compose(basic_transform)


def get_mask_transforms(img_size=256):
    basic_target_transform = T.Compose(
        [
            MaskToTensor(),
            T.Resize(size=(img_size, img_size), interpolation=InterpolationMode.NEAREST),
        ]
    )
    return basic_target_transform


class MaskToTensor:
    def __call__(self, mask):
        return torch.from_numpy(np.array(mask, np.uint8)).unsqueeze(dim=0).float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


##########################################################
# augmentations:
##########################################################

def get_seg_augs(imgz_size=256, data_keys=("input", "mask")):
    default_seg_augs = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomResizedCrop(
            size=(imgz_size, imgz_size), scale=(0.8, 1.0), resample="bilinear", align_corners=False
        ),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
        K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        data_keys=data_keys
    )
    return default_seg_augs


def get_cd_augs(imgz_size=256, data_keys=("input", "input", "mask"), color_jet=False):
    if color_jet:
        default_cd_augs = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            K.RandomResizedCrop(
                size=(imgz_size, imgz_size), scale=(0.8, 1.0), resample="bilinear", align_corners=False
            ),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
            K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            data_keys=data_keys
        )
        print('#################### use color jet augs in the CD task. ####################')
    else:
        default_cd_augs = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomResizedCrop(
                size=(imgz_size, imgz_size), scale=(0.8, 1.0), resample="bilinear", align_corners=False
            ),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
            K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            data_keys=data_keys
        )
    return default_cd_augs


def get_scd_augs(imgz_size=256, data_keys=("input", "input", "mask", "mask")):
    default_scd_augs = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
        K.RandomResizedCrop(
            size=(imgz_size, imgz_size), scale=(0.8, 1.0), resample="bilinear", align_corners=False
        ),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
        K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        data_keys=data_keys
    )
    return default_scd_augs


def get_ssl_augs(img_size=256, data_keys=('input', 'mask')):

    default_ssl_augs = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
        K.RandomResizedCrop(
            size=(img_size, img_size), scale=(0.8, 1.0), resample="bilinear", align_corners=False
            ),
        K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        data_keys=data_keys,
    )
    return default_ssl_augs


def get_ssl_augs_geometry(img_size=256, data_keys=('input', 'mask')):
    default_ssl_augs = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        # K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        # K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
        K.RandomResizedCrop(
            size=(img_size, img_size), scale=(0.8, 1.0), resample="bilinear", align_corners=False
        ),
        K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        data_keys=data_keys,
    )
    return default_ssl_augs


def my_test():
    aug_list = K.AugmentationSequential(
        K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        K.RandomAffine(360, [0.1, 0.1], [0.7, 1.2], [30., 50.], p=1.0),
        K.RandomPerspective(0.5, p=1.0),
        data_keys=["input", "bbox", "keypoints", "mask"],
        return_transform=False,
        same_on_batch=False,
    )

    bbox = torch.tensor([[[355, 10], [660, 10], [660, 250], [355, 250]]])
    keypoints = torch.tensor([[[465, 115], [545, 116]]])


if __name__ == '__main__':
    my_test()