import random
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as TF
from torchvision import transforms, utils
import torch


def to_tensor_and_norm(imgs, labels):
    """
    :param imgs: [ndarray, or PIL data]
    :param labels: [ndarray, or PIL data]
    :return:
    """
    # to tensor
    imgs = [TF.to_tensor(img) for img in imgs]
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
              for img in labels]

    imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            for img in imgs]
    return imgs, labels


class CDDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_brightness=False,
            with_random_blur=False,
            with_random_gamma=False,
            with_random_contrast=False,
            with_random_hue=False,
            with_random_saturation=False,
    ):
        self.img_size = img_size
        #  支持数据集中图像数据存在多种尺度的情况，维持原始尺寸
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.with_random_brightness = with_random_brightness
        self.with_random_contrast = with_random_contrast
        self.with_random_gamma = with_random_gamma
        self.with_random_saturation = with_random_saturation
        self.with_random_hue = with_random_hue

    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = [TF.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]

        labels = [TF.to_pil_image(img) for img in labels]
        if len(labels) != 0:
            if labels[0].size != (self.img_size, self.img_size):
                labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0)
                        for img in labels]

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > random_base:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(img, angle) for img in labels]

        # 数据扩增里面如果也是随机的，那么外面就不用加random()>0.5了，效果更差
        if self.with_random_crop and random.random() > 0:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=imgs[0], scale=(0.8, 1.0), ratio=(1, 1))

            imgs = [TF.resized_crop(img, i, j, h, w,
                                    size=(self.img_size, self.img_size),
                                    interpolation=Image.CUBIC)
                    for img in imgs]

            labels = [TF.resized_crop(img, i, j, h, w,
                                      size=(self.img_size, self.img_size),
                                      interpolation=Image.NEAREST)
                      for img in labels]

        if self.with_scale_random_crop:
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
                    for img in labels]

        if self.with_random_blur and random.random() > 0:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        if self.with_random_brightness and random.random() > random_base:
            # multiply a random number within a - b
            imgs = [TF.adjust_brightness(img, brightness_factor=random.uniform(0.5, 1.5))
                    for img in imgs]

        if self.with_random_gamma and random.random() > random_base:
            # img**gamma
            imgs = [TF.adjust_gamma(img, gamma=random.uniform(0.5, 1.5))
                    for img in imgs]

        if self.with_random_contrast and random.random() > random_base:
            # 0 gives a solid gray image, 1 gives the
            #   original image while 2 increases the contrast by a factor of 2.
            imgs = [TF.adjust_contrast(img, contrast_factor=random.uniform(0.5, 1.5))
                    for img in imgs]

        if self.with_random_hue and random.random() > random_base:
            # both -0.5 and 0.5 will give an image
            # with complementary colors while 0 gives the original image
            imgs = [TF.adjust_hue(img, hue_factor=random.uniform(0, 0.2))
                    for img in imgs]

        if self.with_random_saturation and random.random() > random_base:
            # saturation_factor, 0: grayscale image, 1: unchanged, 2: increae saturation by 2
            imgs = [TF.adjust_saturation(img, saturation_factor=random.uniform(0.5, 1.5))
                    for img in imgs]


        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                      for img in labels]

            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                    for img in imgs]

        return imgs, labels


def pil_crop(image, box, cropsize, default_value):

    assert isinstance(image, Image.Image)
    # if isinstance(image, np.ndarray): images = (images,)
    # if isinstance(image, Image): image = np.array(image)
    # if isinstance(default_values, int): default_values = (default_values,)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype)*default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    """从大小为imgsize的图像中随机裁剪出cropsize大小的图，
    如果裁剪大小小于原图，好理解，裁处部分存在多样性；
    如果裁剪大小小于原图，也能理解，原图的有效部分，在裁剪图的位置也是随机的。因此需要输出两部分
    输出两部分，一是有效图相对裁剪出图的偏移，另一个是裁剪图相对原图的偏移
    out: top, bottom, left, right;
    """
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)
