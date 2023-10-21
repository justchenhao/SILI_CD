from typing import Dict, Optional, Union
import torch
from torch import nn
import kornia as K
import cv2
import torch.nn.functional as F
from einops import rearrange, repeat
import copy


def get_image_edge(image_tensor: torch.Tensor):
    """
    提取batch图像的边缘
    Args:
        image_tensor: b 3 h w / 3 h w (b=1)
    Returns: b h w

    """
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    _, x_canny = K.filters.canny(image_tensor)
    x_canny = x_canny.squeeze(1)
    return x_canny


def get_importance_mask(mask, edge_mask, kernel_size=5):
    """
    去除输入edge_mask中属于mask边缘的部分；防止mask边界处影响mask_push学习
    Args:
        mask: b h w / b 1 h w, float
        edge_mask: b h w, float
    Returns:
    """
    assert edge_mask.ndim == 3
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    assert mask.ndim == 4
    #  获取属于mask前景区域内的edge
    # edge_mask_1 = (mask == 1) * edge_mask
    #  mask 前景边界，重要性降低
    out_importance_mask = copy.deepcopy(edge_mask)

    # kernel_size = 5
    kernel = torch.ones(kernel_size, kernel_size).to(mask.device)
    kernel_l = torch.ones(5, 5).to(mask.device)
    mask_ext = K.morphology.dilation(mask, kernel=kernel_l)
    mask_ine = K.morphology.erosion(mask, kernel=kernel)
    mask_boundary = mask_ext - mask_ine
    mask_boundary = mask_boundary.squeeze(1)

    out_importance_mask[edge_mask == 0] = 1e-5  # 20220415, 非edge区域赋予一般概率，
    out_importance_mask[mask_boundary >= 1] = 1e-8  # mask 边界处赋予低概率
    # from misc.torchutils import visualize_tensors
    # visualize_tensors(edge_mask[0], mask[0], mask_boundary, out_importance_mask[0])
    return out_importance_mask


def sample_points_from_mask(mask: torch.Tensor,
                            target_id: int,
                            num_samples: int,
                            mask_valid: Optional[torch.Tensor] = None,
                            mask_importance: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    在mask上采样值为target_id的点， 返回num_samples个点的(x,y)坐标
    Args:
        mask: Bs * H * W
        target_id:
        num_samples:
        mask_valid:
        mask_importance: Bs * H * W,
    Returns: Bs * num_samples * 2

    """
    if mask.ndim == 4:
        assert mask.shape[1] == 1
        mask = mask.squeeze(dim=1)
    #  如果一个都采样不到怎么办？  TODO:
    h, w = mask.shape[-2], mask.shape[-1]
    sel_mask = (mask == target_id) + 1e-11  # 1e-11 / 1
    if mask_importance is not None:
        assert mask_importance.ndim == 3
        sel_mask = sel_mask * mask_importance
    if mask_valid is not None:
        assert mask_valid.ndim == 3
        sel_mask[mask_valid == 0] = 0  # 0 永远不会被抽到——》multinomial
    # from misc.torchutils import visualize_tensors
    # visualize_tensors(sel_mask[0])
    sel_mask = rearrange(sel_mask, 'b h w -> b (h w)')
    mask_id = torch.multinomial(sel_mask,
                                num_samples=num_samples,
                                replacement=True)  # Bs * num_samples
    sampled_h = mask_id // w
    sampled_w = mask_id % w
    sampled_wh = torch.stack([sampled_w, sampled_h], dim=-1)  # Bs * num_samples * 2
    return sampled_wh


class BiDataAugWithSample(nn.Module):
    """
    分别对前后时相图像做augs，并获取前后时相对应的geometry trans
    利用trans信息，反推原始空间的有效区域，在其中采样，获得points
    """
    def __init__(self, img_size: Union[int, tuple] = 256,
                 num_samples: int = 16,
                 num_classes: int = 2,
                 downsample: int = 32,
                 downsample_init: bool = False,  # 是否在downsample的mask上采样点（可能更分散一点）
                 importance_sample: Union[bool, str] = False,
                 with_scale_ratios: Union[bool, list] = False,
                 ) -> None:
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        assert len(img_size) == 2
        # declare kornia components as class members
        aug_list = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
            K.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            # K.augmentation.RandomAffine(360, [0.1, 0.1], [0.7, 1.2], [30., 50.], p=1.0),
            K.augmentation.RandomResizedCrop(
                size=img_size, scale=(0.8, 1.0), resample="bilinear",
                align_corners=False, cropping_mode="resample",
            ),
            # K.augmentation.RandomPerspective(0.5, p=1.0),
            K.augmentation.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            data_keys=["input", "mask"],
            return_transform=True,
            same_on_batch=False,
        )
        self.augs1 = aug_list
        self.augs2 = copy.deepcopy(aug_list)   # 深拷贝！！，防止id(self.augs1)==id(self.augs2)
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.downsample = downsample
        self.downsample_init = downsample_init
        self.importance_sample = importance_sample
        if self.downsample_init:
            self.pool = nn.AvgPool2d(kernel_size=self.downsample, stride=self.downsample)
        self.with_scale_ratios = with_scale_ratios

    @torch.no_grad()
    def sample_points(self, mask, mask_valid=None, mask_importance=None) -> torch.Tensor:
        """
        根据mask采样num_samples个points
        Returns:

        """
        if self.downsample_init:
            # TODO: 降采样，减少采样点空间过于密集，使用pool，而不是interpolate
            mask = self.pool(mask)  # m会有小数，mask边界处值在【0-1】之间
            mask = (mask > 0.5).float()
            # print(np.unique(m.cpu().numpy()))
            mask_valid = self.pool(mask_valid.float()).long()
            if mask_importance is not None:
                mask_importance = self.pool(mask_importance.float())
        else:
            self.downsample = 1

        out_id = []
        for i in range(self.num_classes):
            sampled_id = sample_points_from_mask(mask, i, self.num_samples,
                                                 mask_valid, mask_importance)
            out_id.append(sampled_id)
        out_id = torch.concat(out_id, dim=1)  # Bs * (num_classes*num_samples) * 2

        out_id = out_id * self.downsample + self.downsample // 2  # 20220414修复
        # TODO: modify, from lr location xy -> hr locations xy
        # out_id = torch.round((out_id + 0.5) * self.downsample - 0.5)
        return out_id

    @torch.no_grad()
    def forward_one(self, x1: torch.Tensor, m1: torch.Tensor):
        # pts = torch.tensor([[[465, 115], [545, 116], [0, 0]]])
        out_tensor = self.augs1(x1, m1)
        x1, trans = out_tensor[0]
        m1 = out_tensor[1]
        return x1, m1

    @torch.no_grad()
    def get_augs_from_predefined(self, x, m):
        """
        根据已有的augs参数，对输入的x做图像变换（几何/颜色）
        @param x: b 3 h w
        @param m: b 1 h w
        @return:
        """
        # https://kornia.readthedocs.io/en/latest/augmentation.container.html
        (x1, trans1), m1 = self.augs1(x, m, params=self.augs1._params, data_keys=["input", "mask"])
        (x2, trans2), m2 = self.augs2(x, m, params=self.augs2._params, data_keys=["input", "mask"])

        return x1, x2, m1, m2

    def _forward_scale_down_up(self, x):
        scales = self.with_scale_ratios
        if isinstance(scales, list):
            scale = scales[torch.randint(len(scales), (1,)).item()]
        else:
            scale = scales
        if scale != 1:
            h, w = x.shape[-2:]
            x = K.geometry.rescale(x, scale)
            # print(x.shape, '``````````````````````````````')
            x = K.geometry.resize(x, (h, w))
        return x
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor,
                m: torch.Tensor,
                pts: Optional[torch.Tensor] = None,
                x_v2: Optional[torch.Tensor] = None,
                importance_mask: Optional[torch.Tensor] = None):
        """

        Args:
            x: b 3 h w
            m: b 1 h w
            pts: None /
            x_v2: None /
            importance_mask: None / b h w, float。
                当importance_sample==True时，importance_mask如果为None，默认由edge方法获得
        Returns:

        """
        if x_v2 is None:
            #  适配多个x输入的情况
            x_v2 = x
        # if m.ndim == 4:
        #     assert m.shape[1] == 1
        #     m = m.squeeze(dim=1)
        # assert m.ndim == 3  # b h w
        h, w = x.shape[-2], x.shape[-1]
        (x1, trans1), m1 = self.augs1(x, m)
        (x2, trans2), m2 = self.augs2(x_v2, m)
        if pts is None:
            x1_, _ = self.augs1.inverse(x1, m1)
            x2_, _ = self.augs2.inverse(x2, m2)
            # self.x1_ = x1_
            # self.x2_ = x2_
            # from misc.torchutils import visualize_tensors
            # visualize_tensors(x, x1_[0], x2_[0])
            # x1_, 非零区域即为可行区域
            mask_valid = (x1_.sum(1) != 0) * (x2_.sum(1) != 0)
            if self.importance_sample:
                if importance_mask is None:
                    edge_mask = get_image_edge(x)
                    importance_mask = get_importance_mask(m, edge_mask)
            pts = self.sample_points(m, mask_valid, importance_mask)
            # pts = check_points_within_boundary(pts, h, w, message='pts')
            self.pts = pts

        pts1 = K.geometry.transform_points(trans1, points_1=pts.float()).long()
        pts2 = K.geometry.transform_points(trans2, points_1=pts.float()).long()
        #  边界条件
        pts1 = check_points_within_boundary(pts1, h, w, message='pts1')
        pts2 = check_points_within_boundary(pts2, h, w, message='pts2')
        if self.with_scale_ratios is not False:
            #TODO: 暂时仅对第二个view做scale aug
            x2 = self._forward_scale_down_up(x2)
            
        return x1, x2, m1, m2, pts1, pts2


def check_torch_greater_than_val(tensor, val):
    import numpy as np
    data = tensor.cpu().numpy()
    index = np.argwhere(data > val)
    if len(index) > 0:
        print(f'data: {tensor[index[0][0],index[0][1]]}, '
              f'index:{index[0][0]},{index[0][1]}')
    return index


def check_torch_greater_equal_than_val(tensor, val):
    import numpy as np
    data = tensor.cpu().numpy()
    index = np.argwhere(data >= val)
    if len(index) > 0:
        print(f'data: {tensor[index[0][0],index[0][1]]}, '
              f'index:{index[0][0]},{index[0][1]}')
    return index


def check_points_within_boundary(pts, h, w, message=''):
    import warnings
    if pts[..., 0].greater_equal(w).sum() > 0:
        warnings.warn(f'{message}: points greater than width {w}')
        # check_torch_greater_equal_than_val(pts[..., 0], w)
        pts[..., 0][pts[..., 0].greater_equal(w)] = w - 1
    if pts[..., 1].greater_equal(h).sum() > 0:
        warnings.warn(f'{message}: points greater than height {h}')
        # check_torch_greater_equal_than_val(pts[..., 0], h)
        pts[..., 1][pts[..., 1].greater_equal(h)] = h - 1
    if (-pts[..., 0]).greater(0).sum() > 0:
        warnings.warn(f'{message}: points smaller than 0')
        # check_torch_greater_than_val(-pts[..., 0], 0)
        pts[..., 0][(-pts[..., 0]).greater(0)] = 0
    if (-pts[..., 1]).greater(0).sum() > 0:
        warnings.warn(f'{message}: points smaller than 0')
        # check_torch_greater_than_val(-pts[..., 0], 0)
        pts[..., 1][(-pts[..., 1]).greater(0)] = 0
    return pts


from torchvision.transforms import transforms
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()
import numpy as np
from misc.torchutils import de_norm


def plot_resulting_image(img, mask, keypoints, radius=2):
    # img = img * mask
    img_draw = np.array(to_pil(img))
    # img_draw = cv2.polylines(np.array(to_pil(img)), bbox.numpy(), isClosed=True, color=(255, 0, 0))
    # for k in keypoints[0]:
    #     img_draw = cv2.circle(img_draw, tuple(k.numpy()[:2]), radius=radius, color=(255, 0, 0), thickness=-1)
    pts_num = len(keypoints[0])
    for i, k in enumerate(keypoints[0]):
        if i < pts_num // 2:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        img_draw = cv2.circle(img_draw, tuple(k.cpu().numpy()[:2]), radius=radius, color=color, thickness=-1)

    return img_draw


def my_test_2():
    from kornia.geometry import bbox_to_mask
    from kornia.utils import image_to_tensor, tensor_to_image
    import matplotlib.pyplot as plt
    from misc.imutils import im2arr, save_image
    from misc.torchutils import seed_torch
    seed_torch(seed=202204)
    img_path = r'samples\LEVIR_cut\B\test_2_0000_0000.png'
    img_path = r'samples\austin10_0512_4096_A.png'
    mask_path = r'samples\LEVIR_cut\label\test_2_0000_0000.png'
    mask_path = r'samples\austin10_0512_4096_label.png'
    img = im2arr(img_path)
    mask = im2arr(mask_path)
    h, w = img.shape[:2]
    img_tensor = image_to_tensor(img).float() / 255.

    mask_tensor = image_to_tensor(mask).float() / 255.

    # pts = torch.tensor([[[465, 115], [545, 116], [0, 0]]])

    aug = BiDataAugWithSample(img_size=(h, w), importance_sample=True, downsample_init=True,
                              downsample=4)
    x1, x2, m1, m2, pts1, pts2 = aug.forward(img_tensor, mask_tensor, pts=None)

    # x1_, x2_ = aug.get_augs_from_predefined(img_tensor)
    # assert x1.mean() == x1_.mean()

    img_out_ori = plot_resulting_image(img_tensor, mask_tensor, aug.pts)

    img_out = plot_resulting_image(
        de_norm(x1[0]),
        m1[0],
        pts1.int(),
    )

    #  down sampled image
    downsample = 4
    pool = nn.AvgPool2d(kernel_size=downsample, stride=downsample)
    x_low = pool(x2)
    m_low = pool(m2)
    # factor = 0.25
    # x_low = F.interpolate(x2, scale_factor=factor)
    # m_low = F.interpolate(m2, scale_factor=factor)
    pts_low = pts2 / downsample
    img_out_low = plot_resulting_image(
        de_norm(x_low[0]),
        m_low[0],
        pts_low.int(),
    )
    fig, axes = plt.subplots(3, 1, figsize=(6, 5))

    axes[0].imshow(img_out_ori)
    axes[0].set_title('original image')
    axes[1].imshow(img_out)
    axes[1].set_title('transformed image 1')
    axes[2].imshow(img_out_low)
    axes[2].set_title('downsampled transformed image 2')
    plt.show()


def draw_sample_vis():
    from kornia.utils import image_to_tensor, tensor_to_image
    import matplotlib.pyplot as plt
    from misc.imutils import im2arr, save_image
    from misc.torchutils import seed_torch
    seed_torch(seed=202204)

    pass


def my_test_canny():
    from kornia.utils import image_to_tensor, tensor_to_image
    import matplotlib.pyplot as plt
    from misc.imutils import im2arr, save_image
    from misc.torchutils import seed_torch
    seed_torch(seed=202204)
    img_path = r'image_transfroms\test_2_0000_0000_B.png'
    mask_path = r'image_transfroms\test_2_0000_0000_label.png'
    img = im2arr(img_path)
    mask = im2arr(mask_path)
    h, w = img.shape[:2]
    img_tensor = image_to_tensor(img).float() / 255.
    mask_tensor = image_to_tensor(mask).float() / 255.

    x_canny = get_image_edge(img_tensor)
    imp_mask = get_importance_mask(mask_tensor, x_canny)

    img_canny: np.ndarray = tensor_to_image(x_canny.byte())
    img_mask = tensor_to_image(imp_mask.byte())
    # Create the plot
    fig, axs = plt.subplots(1, 3, figsize=(16, 16))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].set_title('image source')
    axs[0].imshow(img)

    axs[1].axis('off')
    axs[1].set_title('canny default')
    axs[1].imshow(img_canny, cmap='Greys')
    axs[2].axis('off')
    axs[2].set_title('canny with mask 1')
    axs[2].imshow(img_mask, cmap='Greys')
    plt.show()


def my_test3():
    a = torch.range(1, 64).view([2, 8, 4])
    b = torch.range(1, 64).view([2, 8, 4])
    c = torch.concat([a, b], dim=1)
    from einops import rearrange
    c2 = rearrange(c, 'b (nc ns) n -> b nc ns n', nc=2)
    print((c2[:, 0]-a).sum())
    print((c2[:, 1]-b).sum())



if __name__ == '__main__':
    my_test_2()
    # my_test_canny()