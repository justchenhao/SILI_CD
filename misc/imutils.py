import random
import numpy as np
import cv2
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import unary_from_labels
from PIL import Image
from PIL import ImageFilter
from skimage import io


# 旋转angle角度，缺失背景borderValue填充
def cv_rotate(image, angle, borderValue):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    if isinstance(borderValue, int):
        values = (borderValue, borderValue, borderValue)
    else:
        values = borderValue
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=values)


def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST

    return np.asarray(Image.fromarray(img).resize(size[::-1], resample))


def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(height*scale), int(width*scale))
    return pil_resize(img, target_size, order)


def pil_downup_scale(img, scale, order):
    height, width = img.shape[:2]
    img = pil_rescale(img, scale, order)
    img = pil_resize(img, [height, width], order)
    return img


def random_crop_area_downup_scale(img, scale, cropsize=128, order=3):
    height, width = img.shape[:2]
    box = get_random_crop_box(imgsize=[height, width], cropsize=cropsize)
    img_ = img.copy()
    img_[box[4]:box[5], box[6]:box[7]] = pil_downup_scale(img[box[4]:box[5], box[6]:box[7]], scale, order)

    return img_

import torch


def patch4_random_scale(img, scales, order=3):
    height, width = img.shape[:2]
    crop_size = height // 2
    img_ = img.copy()
    for x1 in [0, width//2]:
        for y1 in [0, height//2]:
            scale = scales[torch.randint(len(scales), (1,)).item()]
            img_[y1:y1+crop_size, x1:x1+crop_size] = pil_downup_scale(img[y1:y1+crop_size, x1:x1+crop_size], scale=scale, order=order)
    return img_


def random_crop_exchange(img1, img2, crop_size=128):
    # 随机交换两个时相图的的cropsize区域
    height, width = img1.shape[:2]
    box = get_random_crop_box(imgsize=[height, width], cropsize=crop_size)
    img1_ = img1.copy()
    img2_ = img2.copy()
    img1_[box[4]:box[5], box[6]:box[7]] = img2[box[4]:box[5], box[6]:box[7]]
    img2_[box[4]:box[5], box[6]:box[7]] = img1[box[4]:box[5], box[6]:box[7]]
    return img1_, img2_


def pil_rotate(img, degree, default_value):
    if isinstance(default_value, tuple):
        values = (default_value[0], default_value[1], default_value[2], 0)
    else:
        values = (default_value, default_value, default_value,0)
    img = Image.fromarray(img)
    if img.mode =='RGB':
        # set img padding == default_value
        img2 = img.convert('RGBA')
        rot = img2.rotate(degree, expand=1)
        fff = Image.new('RGBA', rot.size, values)  # 灰色
        out = Image.composite(rot, fff, rot)
        img = out.convert(img.mode)

    else:
        # set label padding == default_value
        img2 = img.convert('RGBA')
        rot = img2.rotate(degree, expand=1)
        # a white image same size as rotated image
        fff = Image.new('RGBA', rot.size, values)
        # create a composite image using the alpha layer of rot as a mask
        out = Image.composite(rot, fff, rot)
        img = out.convert(img.mode)

    return np.asarray(img)


def random_resize_long_image_list(img_list, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h, w = img_list[0].shape[:2]
    if w < h:
        scale = target_long / h
    else:
        scale = target_long / w
    out = []
    for img in img_list:
        out.append(pil_rescale(img, scale, 3) )
    return out

def random_resize_long(img, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h, w = img.shape[:2]

    if w < h:
        scale = target_long / h
    else:
        scale = target_long / w

    return pil_rescale(img, scale, 3)


def random_scale_list(img_list, scale_range, order):
    """
        输入：图像列表
    """
    target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    if isinstance(img_list, tuple):
        assert img_list.__len__() == 2
        img1 = []
        img2 = []
        for img in img_list[0]:
            img1.append(pil_rescale(img, target_scale, order[0]))
        for img in img_list[1]:
            img2.append(pil_rescale(img, target_scale, order[1]))
        return (img1, img2)
    else:
        out = []
        for img in img_list:
            out.append(pil_rescale(img, target_scale, order))
        return out

def random_scale(img, scale_range, order):

    target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    if isinstance(img, tuple):
        return (pil_rescale(img[0], target_scale, order[0]), pil_rescale(img[1], target_scale, order[1]))
    else:
        return pil_rescale(img, target_scale, order)


def random_rotate_list(img_list, max_degree, default_values):
    degree = random.random() * max_degree
    if isinstance(img_list, tuple):
        assert img_list.__len__() == 2
        img1 = []
        img2 = []
        for img in img_list[0]:
            assert isinstance(img, np.ndarray)
            img1.append((pil_rotate(img, degree, default_values[0])))
        for img in img_list[1]:
            img2.append((pil_rotate(img, degree, default_values[1])))
        return (img1, img2)
    else:
        out = []
        for img in img_list:
            out.append(pil_rotate(img, degree, default_values))
        return out


def random_rotate(img, max_degree, default_values):
    degree = random.random() * max_degree
    if isinstance(img, tuple):
        return (pil_rotate(img[0], degree, default_values[0]),
                pil_rotate(img[1], degree, default_values[1]))
    else:
        return pil_rotate(img, degree, default_values)


def random_lr_flip_list(img_list):

    if bool(random.getrandbits(1)):
        if isinstance(img_list, tuple):
            assert img_list.__len__()==2
            img1=list((np.fliplr(m) for m in img_list[0]))
            img2=list((np.fliplr(m) for m in img_list[1]))

            return (img1, img2)
        else:
            return list([np.fliplr(m) for m in img_list])
    else:
        return img_list


def random_lr_flip(img):

    if bool(random.getrandbits(1)):
        if isinstance(img, tuple):
            return tuple([np.fliplr(m) for m in img])
        else:
            return np.fliplr(img)
    else:
        return img

def get_random_crop_box(imgsize, cropsize):
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


def random_crop_list(images_list, cropsize, default_values):

    if isinstance(images_list, tuple):
        imgsize = images_list[0][0].shape[:2]
    elif isinstance(images_list, list):
        imgsize = images_list[0].shape[:2]
    else:
        raise RuntimeError('do not support the type of image_list')
    if isinstance(default_values, int): default_values = (default_values,)

    box = get_random_crop_box(imgsize, cropsize)
    if isinstance(images_list, tuple):
        assert images_list.__len__()==2
        img1 = []
        img2 = []
        for img in images_list[0]:
            f = default_values[0]
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            img1.append(cont)
        for img in images_list[1]:
            f = default_values[1]
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            img2.append(cont)
        return (img1, img2)
    else:
        out = []
        for img in images_list:
            f = default_values
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype) * f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            out.append(cont)
        return out


def random_crop(images, cropsize, default_values):

    if isinstance(images, np.ndarray): images = (images,)
    if isinstance(default_values, int): default_values = (default_values,)

    imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, default_values):

        if len(img.shape) == 3:
            cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
        else:
            cont = np.ones((cropsize, cropsize), img.dtype)*f
        cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
        new_images.append(cont)

    if len(new_images) == 1:
        new_images = new_images[0]

    return new_images

def top_left_crop(img, cropsize, default_value):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[:ch, :cw] = img[:ch, :cw]

    return container

def center_crop(img, cropsize, default_value=0):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    sh = h - cropsize
    sw = w - cropsize

    if sw > 0:
        cont_left = 0
        img_left = int(round(sw / 2))
    else:
        cont_left = int(round(-sw / 2))
        img_left = 0

    if sh > 0:
        cont_top = 0
        img_top = int(round(sh / 2))
    else:
        cont_top = int(round(-sh / 2))
        img_top = 0

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        img[img_top:img_top+ch, img_left:img_left+cw]

    return container


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))

def pil_blur(img, radius):
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius)))


def random_blur(img):
    radius = random.random()
    # print('add blur: ', radius)
    if isinstance(img, list):
        out = []
        for im in img:
            out.append(pil_blur(im, radius))
        return out
    elif isinstance(img, np.ndarray):
        return pil_blur(img, radius)
    else:
        print(img)
        raise RuntimeError("do not support the input image type!")


from torchvision import transforms
def torch_colorjit(img, brightness=0.4, contrast=1, saturation=1, hue=0.1):
    colorJitter = transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                         saturation=saturation, hue=hue)

    return np.array(colorJitter(Image.fromarray(img)))

def random_jit(img):
    brightness = 0.4
    contrast = 1
    saturation = 1
    hue = 0.1
    if isinstance(img, list):
        out = []
        for im in img:
            out.append(torch_colorjit(im, brightness=brightness,
                                      contrast=contrast, saturation=saturation, hue=hue))
        return out
    elif isinstance(img, np.ndarray):
        return torch_colorjit(img, brightness=brightness,
                                      contrast=contrast, saturation=saturation, hue=hue)
    else:
        print(img)
        raise RuntimeError("do not support the input image type!")


def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)


def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride


def compress_range(arr):
    uniques = np.unique(arr)
    maximum = np.max(uniques)

    d = np.zeros(maximum+1, np.int32)
    d[uniques] = np.arange(uniques.shape[0])

    out = d[arr]
    return out - np.min(out)


def colorize_score(score_map, exclude_zero=False, normalize=True, by_hue=False):
    import matplotlib.colors
    if by_hue:
        aranged = np.arange(score_map.shape[0]) / (score_map.shape[0])
        hsv_color = np.stack((aranged, np.ones_like(aranged), np.ones_like(aranged)), axis=-1)
        rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)

        test = rgb_color[np.argmax(score_map, axis=0)]
        test = np.expand_dims(np.max(score_map, axis=0), axis=-1) * test

        if normalize:
            return test / (np.max(test) + 1e-5)
        else:
            return test

    else:
        VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                     (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                     (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                     (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)

        if exclude_zero:
            VOC_color = VOC_color[1:]

        test = VOC_color[np.argmax(score_map, axis=0)%22]
        test = np.expand_dims(np.max(score_map, axis=0), axis=-1) * test
        if normalize:
            test /= np.max(test) + 1e-5

        return test


def colorize_displacement(disp):

    import matplotlib.colors
    import math

    a = (np.arctan2(-disp[0], -disp[1]) / math.pi + 1) / 2

    r = np.sqrt(disp[0] ** 2 + disp[1] ** 2)
    s = r / np.max(r)
    hsv_color = np.stack((a, s, np.ones_like(a)), axis=-1)
    rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)

    return rgb_color


def colorize_label(label_map, normalize=True, by_hue=True, exclude_zero=False, outline=False):

    label_map = label_map.astype(np.uint8)

    if by_hue:
        import matplotlib.colors
        sz = np.max(label_map)
        aranged = np.arange(sz) / sz
        hsv_color = np.stack((aranged, np.ones_like(aranged), np.ones_like(aranged)), axis=-1)
        rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)
        rgb_color = np.concatenate([np.zeros((1, 3)), rgb_color], axis=0)

        test = rgb_color[label_map]
    else:
        VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                              (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                              (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                              (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)

        if exclude_zero:
            VOC_color = VOC_color[1:]
        test = VOC_color[label_map]
        if normalize:
            test /= np.max(test)

    if outline:
        edge = np.greater(np.sum(np.abs(test[:-1, :-1] - test[1:, :-1]), axis=-1) + np.sum(np.abs(test[:-1, :-1] - test[:-1, 1:]), axis=-1), 0)
        edge1 = np.pad(edge, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        edge2 = np.pad(edge, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        edge = np.repeat(np.expand_dims(np.maximum(edge1, edge2), -1), 3, axis=-1)

        test = np.maximum(test, edge)
    return test


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(np.array(image_numpy,dtype=np.uint8))
    image_pil.save(image_path)

import PIL
import skimage
import tifffile
import cv2 as cv

def im2arr(img_path, mode=1, dtype=np.uint8):
    """
    :param img_path:
    :param mode:
    :return: numpy.ndarray, shape: H*W*C
    """
    if mode==1:
        img = PIL.Image.open(img_path)
        arr = np.asarray(img, dtype=dtype)
    elif mode==2:
        arr = skimage.io.imread(img_path)
        arr = arr.astype(dtype)
    elif mode == 4:
        arr = cv.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
        arr = arr.astype(dtype)
    else:
        arr = tifffile.imread(img_path)
        if arr.ndim == 3:
            a, b, c = arr.shape
            if a < b and a < c:  # 当arr为C*H*W时，需要交换通道顺序
                arr = arr.transpose([1,2,0])
    # print('shape: ', arr.shape, 'dytpe: ',arr.dtype)
    return arr

def get_connetcted_info(L):
    """利用opencv联通域分析方法
    返回连通域数量，连通域图，各个连通域的统计信息（右上角点坐标x/y/长/宽/像素面积）
    :param L: ndarray： H*W*3
    :return: num_labels(int), labels(ndarray：H*W), stats(list)
    stats[i]: x1, y1, dx, dy, area
    """
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))  # 定义矩形结构元素
    #
    # L = cv.morphologyEx(L, cv.MORPH_OPEN, kernel, iterations=1)  # 开运算1

    num_labels, labels, stats, centers = cv.connectedComponentsWithStats(L, connectivity=8, ltype=cv.CV_32S)
    # print("there are ", num_labels - 1, ' instances.')

    # # 从第1类开始，第零类是背景
    # for i in range(1, num_labels):
    #     mask_tmp = (labels==i)
    # x1, y1, dx, dy, area = stats[i]  # x1, y1, x2, y2, area
    return num_labels, labels, stats


def get_center_mask(mask):
    """返回mask中心的目标区域mask
    mask：ndarray，3*H*W
    """
    h,w = mask.shape[:2]
    # 联通域分析
    num_labels, labels, stats = get_connetcted_info(mask)
    center_label = labels[h//2,w//2]
    #  如果中心mask的中心点不属于该mask（凹图形）
    if center_label == 0:
        # 找到中心接近图像中心的建筑区域。
        for i in range(1, num_labels):
            x1, y1, dx, dy, area = stats[i]
            if (abs(x1+dx//2-w//2) <= 2) or (abs(y1+dy//2-h//2)<=2):
                center_label = i
                # print(x1+dx//2, w//2)
                # print(y1 + dy // 2, h // 2)
                break
    out = np.array(labels==center_label,dtype=np.uint8)
    return out

def mask_colorize(label_mask):
    """
    :param label_mask: mask (np.ndarray): (M, N), uint8
    :return: color label: (M, N, 3), uint8
    """
    assert isinstance(label_mask, np.ndarray)
    assert label_mask.ndim == 2
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.float)
    r = label_mask % 6
    g = (label_mask % 36) // 6
    b = label_mask // 36
    # 归一化到[0-1]
    rgb[:, :, 0] = r / 6
    rgb[:, :, 1] = g / 6
    rgb[:, :, 2] = b / 6
    rgb = np.array(rgb * 255, dtype=np.uint8)
    return rgb


# colour map
label_colours = [(0, 0, 0)
                 # 0=background
    , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
    , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]


# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

def mask_colorize(mask, label_colours=label_colours):
    """
    按照指定索引给mask上色
    :param mask:
    :param label_colours:
    :param num_classes:
    :return:
    """
    assert isinstance(mask, np.ndarray)
    assert mask.ndim == 2
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    max_num = mask.max()
    # 从0-max_num染色
    for i in range(max_num+1):

        output[mask==i] = label_colours[i]

    return output


def apply_colormap(norm_map, colormode='jet'):
    """
    输入：归一化的图，{[0,1]}^(H*W)；
    输出：上色后的图。
    """
    assert norm_map.ndim == 2

    if colormode == 'jet':
        colormap = cv.COLORMAP_JET
    elif colormode == 'twilight':
        colormap = cv.COLORMAP_TWILIGHT
    elif colormode == 'rainbow':
        colormap = cv.COLORMAP_RAINBOW
    else:
        raise NotImplementedError

    norm_map_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8),
                                       colormap=colormap)

    norm_map_color = norm_map_color[..., ::-1]

    return norm_map_color


def concat_imgs(*images,padding=5):
    # padding = 5
    nums = len(images)
    h, w = images[0].shape[:2]
    out = np.ones([h, (w+padding)*nums, 3], dtype=np.uint8)*255

    for i, image in enumerate(images):
        if image.ndim == 2:
            image = np.concatenate([image[...,np.newaxis]]*3, axis=-1)
        out[:,(w+padding)*i:(w+padding)*i+w] = image

    return out


def concat_img_horizon(left_img, right_img, padding=5):
    h, w = left_img.shape[:2]
    h2,w2 = right_img.shape[:2]
    target_w2 = round(h/h2*w2)
    right_img = pil_resize(right_img, (h,target_w2),order=3)
    out = np.ones([h, w+target_w2+padding, 3], dtype=np.uint8)*255
    out[:,:w] = left_img
    out[:,w+padding:w+padding+target_w2] = right_img
    return out


def concat_img_vertical(up_img, down_img, padding=5):
    # padding = 5
    h, w = up_img.shape[:2]
    out = np.ones([h*2+padding, w, 3], dtype=np.uint8)*255
    for i, image in enumerate([up_img, down_img]):
        if image.ndim == 2:
            image = np.concatenate([image[...,np.newaxis]]*3, axis=-1)
        out[(h+padding)*i:(h+padding)*i+h] = image
    return out
