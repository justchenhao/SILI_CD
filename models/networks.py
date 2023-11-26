from typing import Union
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import os
import math
import numpy as np
from kornia.filters import canny, sobel
from functools import partial

from models import resnet


# Decide which device we want to run on
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02,
                 ignore_prefix='backbone', Info=True):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    if Info:
        print('initialize network with %s' % init_type)
    # for name, model in net.named_modules(): # 这个有多层结构的，不行。。。
    for name, model in net.named_children():
        if ignore_prefix not in name:
            if Info:
                print('model init: %s' % name)
            # print(model)
            model.apply(init_func)
    # net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, init_type, init_gain=init_gain)
    #TODO: update to torch2.0
    # torch_version = torch.__version__
    # if torch_version.split('.')[0] == '2' and torch.cuda.get_device_capability()[0] >= 7.0:
    #     net = torch.compile(net)
    #     print(f'using: {torch_version}~~~~~~~~~')
    return net


class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )


class ThreeLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels // 2, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=kernel_size,
                                   padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels // 2),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )


class FCN_Decoder(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), backbone_stages_num=5):
        super(FCN_Decoder, self).__init__()
        self.backbone_stages_num = backbone_stages_num
        if self.backbone_stages_num == 5:
            self.conv_fpn1 = nn.Conv2d(channels[-1],
                                       channels[-2], kernel_size=3, padding=1)
        if self.backbone_stages_num >= 4:
            self.conv_fpn2 = nn.Conv2d(channels[-2], channels[-3], kernel_size=3, padding=1)
        self.conv_fpn3 = nn.Conv2d(channels[-3], channels[-4], kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)


    def forward(self, feats):
        if self.backbone_stages_num == 5:
            x_4, x_8, x_16, x_32 = feats[1:]
                # FPN layers
            x = self.upsamplex2(self.relu(self.conv_fpn1(x_32)))
            x = self.upsamplex2(self.relu(self.conv_fpn2(x + x_16)))
            x = self.upsamplex2(self.relu(self.conv_fpn3(x + x_8)))
        elif self.backbone_stages_num == 4:
            x_4, x_8, x_16 = feats[1:]
            # FPN layers
            x = self.upsamplex2(self.relu(self.conv_fpn2(x_16)))
            x = self.upsamplex2(self.relu(self.conv_fpn3(x + x_8)))
        else:
            raise NotImplementedError(self.resnet_stages_num)
        return x


def canny_channelwise(x):
    b, c, h, w = x.shape
    from einops import rearrange
    x = rearrange(x, 'b c h w->(b c) 1 h w')
    x = canny(x)[0]
    x = rearrange(x, '(b c) 1 h w->b c h w', b=b)
    return x


class EdgeBtNeck(nn.Module):
    """
    对前后时相图像计算边缘图
    """
    def __init__(self, out_size=None, align_corners=True, with_learnable_edge=False,
                 edge_dim=3, with_edge_type='sobel', mode='simple') -> None:
        super(EdgeBtNeck, self).__init__()

        self.out_size = out_size
        self.align_corners = align_corners
        self.with_learnable_edge = with_learnable_edge
        self.with_edge_type = with_edge_type
        self.mode = mode
        if self.mode == 'conv_concat':
            # self.conv = TwoLayerConv2d(in_channels=6, out_channels=edge_dim, kernel_size=3)
            self.conv = nn.Conv2d(in_channels=6, out_channels=edge_dim, kernel_size=(7, 7))
        elif self.mode == 'filter_conv':
            self.conv = nn.Conv2d(in_channels=3, out_channels=edge_dim, kernel_size=(7, 7))

        if self.with_edge_type == 'canny':
            self.filter = canny_channelwise
        elif self.with_edge_type == 'no':
            self.filter = nn.Identity()
        else:
            self.filter = partial(sobel, normalized=True, eps=1e-06)

    def forward(self, img, img2):
        out_size = self.out_size if self.out_size is not None else img.shape[2:]
        if self.mode == 'conv_concat':
            x_concat = torch.concat([img, img2], dim=1)
            x = self.conv(x_concat)
            x = self.filter(x)
            if isinstance(x, tuple):
                x = x[0]
            if out_size[0] != x.shape[-2]:
                x = F.interpolate(x, size=out_size, mode='bilinear',
                            align_corners=self.align_corners)
        else:
            xs = []
            for x in [img, img2]:
                x_ = self.filter(x)
                if isinstance(x_, tuple):
                    x_ = x_[0]
                # from misc.torchutils import visualize_tensors
                # visualize_tensors(x_[0][0])
                # visualize_tensors(xs[1][0][0])
                # x_ = x_ / 255  # 归一化到[0,1]
                xs.append(x_)
            if out_size[0] != xs[0].shape[-2]:
                for i in range(len(xs)):
                    xs[i] = F.interpolate(xs[i], size=out_size, mode='bilinear',
                            align_corners=self.align_corners)
            x = xs[0] + xs[1]
            if self.mode == 'filter_conv':
                x = self.conv(x)
        return x


##################################################################################
def get_backbone(backbone_name, pretrained=True, backbone_stages_num=5,
                 structure='simple3', ret_mid_layer_channels=False):
    if structure == 'simple3':
        #  表示backbone输出下采样2^3倍
        replace_stride_with_dilation = [False, True, True]
    elif structure == 'simple4':
        #  表示backbone输出下采样2^4倍
        replace_stride_with_dilation = [False, False, True]
    else:
        #  simple5
        #  表示backbone输出下采样2^5倍
        replace_stride_with_dilation = [False, False, False]
    # elif structure == 'fcn' or structure == 'unet' or structure == 'bifpn' :
    #     replace_stride_with_dilation = [False, False, False]
    out_layer_n = [1, ] * 5
    if 'resnet' in backbone_name:
        expand = 1
        if backbone_name == 'resnet18':
            backbone = resnet.resnet18(pretrained=pretrained,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
        elif backbone_name == 'resnet34':
            backbone = resnet.resnet34(pretrained=pretrained,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
        elif backbone_name == 'resnet50':
            backbone = resnet.resnet50(pretrained=pretrained,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
            expand = 4
        else:
            raise NotImplementedError
        out_layer_n = [64, 64, 128, 256, 512]
        out_layer_n = [i * expand for i in out_layer_n]
        out_layer_n = out_layer_n[:backbone_stages_num]
        assert backbone_stages_num >= 3
    else:
        import timm
        model_names = timm.list_models(pretrained=True)
        if backbone_name in model_names:
            backbone = timm.create_model(backbone_name, features_only=True,
                                         out_indices=(0, 1, 2, 3, 4), pretrained=True)
            #TODO whether true
            out_layer_n = backbone.feature_info.channels()[:backbone_stages_num]
        else:
            raise NotImplementedError()
    if ret_mid_layer_channels:
        return backbone, out_layer_n
    else:
        return backbone, out_layer_n[-1]


class Backbone(torch.nn.Module):
    """
    # 当with_out_conv==True时，output_nc无效，与encoder输出的维度有关
    """
    def __init__(self, input_nc, output_nc,
                 backbone_stages_num=5, backbone_name='resnet18',backbone_stage_start_id=0,
                 pretrained=True,
                 head_pretrained=False,
                 frozen_backbone_weights=False,
                 structure='simple3', backbone_fuse_mode='add',
                 with_out_conv=True, out_upsample2x=False,
                 backbone_out_feats=False,
                 scale_aware=False, scale_levels=(1,2,3,4,5)):
        super(Backbone, self).__init__()
        self.backbone_stages_num = backbone_stages_num
        self.backbone_stage_start_id = backbone_stage_start_id
        self.backbone_name = backbone_name
        self.resnet, layers = get_backbone(backbone_name, pretrained=pretrained,
                                           backbone_stages_num=backbone_stages_num,
                                           structure=structure)
        self.structure = structure
        self.head_pretrained = head_pretrained
        self.with_out_conv = with_out_conv
        self.out_upsample2x = out_upsample2x
        if self.structure == 'fcn':
            assert 'resnet' in self.backbone_name
            self.expand = 1
            if self.backbone_name == 'resnet50':
                self.expand = 4
            if self.backbone_stages_num == 5:
                self.conv_fpn1 = nn.Conv2d(512 * self.expand,
                                           256 * self.expand, kernel_size=3, padding=1)
            if self.backbone_stages_num >= 4:
                self.conv_fpn2 = nn.Conv2d(256 * self.expand, 128 * self.expand, kernel_size=3, padding=1)

            self.conv_fpn3 = nn.Conv2d(128 * self.expand, 64 * self.expand, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.upsamplex2 = nn.Upsample(scale_factor=2)
            layers = 64 * self.expand
        elif self.structure == 'unet':
            assert 'resnet' in self.backbone_name
            self.expand = 1
            if self.backbone_name == 'resnet50':
                self.expand = 4
            if self.backbone_stages_num >= 5:
                self.conv_fpn1 = nn.Conv2d(512 * self.expand, 256 * self.expand, kernel_size=3, padding=1)
                self.conv_fpn2 = nn.Conv2d(256 * 2 * self.expand, 128 * self.expand, kernel_size=3, padding=1)
                self.conv_fpn3 = nn.Conv2d(128 * 2 * self.expand, 64 * self.expand, kernel_size=3, padding=1)
                self.conv_fpn4 = nn.Conv2d(64 * 2 * self.expand, 64 * self.expand, kernel_size=3, padding=1)
                self.conv_fpn5 = nn.Conv2d(64 * 2 * self.expand, 32 * self.expand, kernel_size=3, padding=1)
            elif self.backbone_stages_num == 4:
                self.conv_fpn2 = nn.Conv2d(256 * self.expand, 128 * self.expand, kernel_size=3, padding=1)
                self.conv_fpn3 = nn.Conv2d(128 * 2 * self.expand, 64 * self.expand, kernel_size=3, padding=1)
                self.conv_fpn4 = nn.Conv2d(64 * 2 * self.expand, 64 * self.expand, kernel_size=3, padding=1)
                self.conv_fpn5 = nn.Conv2d(64 * 2 * self.expand, 32 * self.expand, kernel_size=3, padding=1)
            else:
                raise NotImplementedError
            self.relu = nn.ReLU()
            self.upsamplex2 = nn.Upsample(scale_factor=2)
            layers = 32 * self.expand

        elif self.structure == 'fpn':
            from models.FPN import FPNDecoder
            # TODO: 暂时只支持resent18
            assert 'resnet18' in backbone_name
            encoder_channels = [64, 128, 256, 512][::-1]
            pyramid_channels = 256
            self.neck = FPNDecoder(encoder_channels=encoder_channels, encoder_depth=5,
                            pyramid_channels=pyramid_channels, with_segmentation_head=False)
            layers = pyramid_channels
        elif 'simple' not in self.structure:
            raise NotImplementedError
        self.out_layers = layers
        if self.with_out_conv:
            self.conv_pred = nn.Conv2d(layers, output_nc, kernel_size=3, padding=1)
            self.out_layers = output_nc
        if self.out_upsample2x:
            self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.backbone_out_feats = backbone_out_feats
        if self.backbone_out_feats:
            assert 'simple' in structure
            assert out_upsample2x is False
            assert with_out_conv is False
            # P1, p2, p3, p4, p5
            self.out_feat_channels = [64, 64, 128, 256, 512]
        self.scale_aware = scale_aware

        init_weights(self, ignore_prefix='resnet')
        self._load_backbone_pretrain(pretrained)
        if frozen_backbone_weights:
            self._frozen_backbone()

    def _make_stages(self):
        if 'resnet' in self.backbone_name:
            stages = self.backbone.get_stages()
        else:
            raise NotImplementedError
        return stages[self.backbone_stage_start_id:self.backbone_stages_num]

    def _load_backbone_pretrain(self, pretrained):
        """用于加载预训练模型，用于下游迁移训练,
        以及是否迁移head 参数"""

        if pretrained is not None and os.path.isfile(pretrained):
            assert os.path.exists(pretrained)
            state_dict = torch.load(pretrained)
            loaded_state_dict = {}
            for k, v in state_dict.items():
                if 'resnet.' in k:
                    loaded_state_dict[k] = state_dict[k]
                else:
                    if self.head_pretrained:
                        loaded_state_dict[k] = state_dict[k]
            if not self.head_pretrained:
                print('do not loaded head parameter from %s' % pretrained)
            try:
                print(f'loading pretrained with items: {loaded_state_dict.keys()}')
                self.load_state_dict(loaded_state_dict, strict=False)
            except RuntimeError as e:
                print(e)
                # params_model = list(self.resnet.conv1.named_parameters())
                # params = state_dict['resnet.conv1.weight']
                # print('%s, shape: %s;[0][0][0]: %s' % (params_model[0][0],
                #                                        params_model[0][1].shape,
                #                                        params_model[0][1][0][0][0]))
                # print('%s shape: %s;[0][0][0]: %s' % ('resnet.conv1.weight',
                #                                       params.shape, params[0][0][0]))

            print('Backbone --> load from pretrain: %s' % pretrained)
        else:
            print('Backbone init: %s' % pretrained)

    def _frozen_backbone(self, frozen_layers='resnet'):
        if frozen_layers == 'resnet':
            m = self.resnet
            [x.requires_grad_(False) for x in m.parameters()]
            print(f'frozen resnet pretrained weights')

    def _forward_backbone(self, x):
        if 'resnet' in self.backbone_name:
            x = self.forward_resnet(x)
        else:
            x = self.resnet(x)
        return x

    def forward(self, x, with_feat=False, scale=1):
        if self.scale_aware:
            feats = self._forward_resnet_with_feats_scale(x, scale)  # x_2, x_4, x_8, x_16, x_32
        else:
            feats = self._forward_resnet_with_feats(x)  # x_2, x_4, x_8, x_16, x_32
        if 'simple' in self.structure:
            if self.backbone_out_feats:
                x = feats
            else:
                # TODO: temp only for resnet
                x = feats[0]
        elif self.structure == 'fcn':
            x = self.forward_resnet_fcn(x, feats=feats)
        elif self.structure == 'unet':
            x = self.forward_resnet_unet(x, feats=feats)
        elif self.structure == 'fpn':
            feats = feats[::-1]  # p5,p4,p3,p2
            x = self.neck(feats)
        if self.with_out_conv:
            x = self.conv_pred(x)
        if self.out_upsample2x:
            x = self.upsamplex2(x)
        if with_feat:
            return x, feats
        return x

    # def forward(self, x):
    #     if 'simple' in self.structure:
    #         if self.backbone_out_feats:
    #             x = self._forward_resnet_with_feats(x)[::-1][:4]  # p5,p4,p3,p2
    #         else:
    #             x = self._forward_backbone(x)
    #     elif self.structure == 'fcn':
    #         x = self.forward_resnet_fcn(x)
    #     elif self.structure == 'unet':
    #         x = self.forward_resnet_unet(x)
    #     elif self.structure == 'fpn':
    #         feats = self._forward_resnet_with_feats(x)[::-1]  # p5,p4,p3,p2
    #         x = self.neck(feats)
    #     if self.with_out_conv:
    #         x = self.conv_pred(x)
    #     if self.out_upsample2x:
    #         x = self.upsamplex2(x)
    #     return x

    def forward_resnet(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128

        if self.backbone_stages_num >= 4:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256
        if self.backbone_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/32, in=256, out=512
        elif self.backbone_stages_num > 5:
            raise NotImplementedError
        return x_8

    def _forward_resnet_with_feats(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x_2 = self.resnet.relu(x)
        x = self.resnet.maxpool(x_2)
        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128
        x_16 = self.resnet.layer3(x_8)  # 1/16, in=128, out=256
        if self.backbone_stages_num == 5:
            x_32 = self.resnet.layer4(x_16)  # 1/32, in=256, out=512
            return x_2, x_4, x_8, x_16, x_32
        elif self.backbone_stages_num == 4:
            return x_2, x_4, x_8, x_16
        else:
            raise NotImplementedError

    def _forward_resnet_with_feats_scale(self, x, scale=1):
        # resnet layers
        level_id = 0
        if 0 in self.scale_levels:
            x = self.scale_adapters[level_id](x, scale)
            level_id += 1
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x_2 = self.resnet.relu(x)
        if 1 in self.scale_levels:
            x_2 = self.scale_adapters[level_id](x_2, scale)
            level_id += 1
        x = self.resnet.maxpool(x_2)
        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        if 2 in self.scale_levels:
            x_4 = self.scale_adapters[level_id](x_4, scale)
            level_id += 1
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128
        if 3 in self.scale_levels:
            x_8 = self.scale_adapters[level_id](x_8, scale)
            level_id += 1
        x_16 = self.resnet.layer3(x_8)  # 1/16, in=128, out=256
        if 4 in self.scale_levels:
            x_16 = self.scale_adapters[level_id](x_16, scale)
            level_id += 1
        if self.backbone_stages_num == 5:
            x_32 = self.resnet.layer4(x_16)  # 1/32, in=256, out=512
            if 5 in self.scale_levels:
                x_32 = self.scale_adapters[level_id](x_32, scale)
            return x_2, x_4, x_8, x_16, x_32
        elif self.backbone_stages_num == 4:
            return x_2, x_4, x_8, x_16
        else:
            raise NotImplementedError

    def forward_resnet_fcn(self, x, feats=None):
        if self.backbone_stages_num == 5:
            if feats is None:
                x_4, x_8, x_16, x_32 = self._forward_resnet_with_feats(x)[1:]
            else:
                x_4, x_8, x_16, x_32 = feats[1:]
                # FPN layers
            x = self.upsamplex2(self.relu(self.conv_fpn1(x_32)))
            x = self.upsamplex2(self.relu(self.conv_fpn2(x + x_16)))
            x = self.upsamplex2(self.relu(self.conv_fpn3(x + x_8)))
        elif self.backbone_stages_num == 4:
            if feats is None:
                x_4, x_8, x_16 = self._forward_resnet_with_feats(x)[1:]
            else:
                x_4, x_8, x_16 = feats[1:]
            # FPN layers
            x = self.upsamplex2(self.relu(self.conv_fpn2(x_16)))
            x = self.upsamplex2(self.relu(self.conv_fpn3(x + x_8)))
        else:
            raise NotImplementedError(self.resnet_stages_num)
        return x

    def forward_resnet_unet(self, x, feats=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if self.backbone_stages_num == 5:
            if feats is None:
                x_2, x_4, x_8, x_16, x_32 = self._forward_resnet_with_feats(x)
            else:
                x_2, x_4, x_8, x_16, x_32 = feats
            # FPN layers
            x = self.upsamplex2(self.relu(self.conv_fpn1(x_32)))
            x = torch.cat([x, x_16], dim=1)
            x = self.upsamplex2(self.relu(self.conv_fpn2(x)))
            x = torch.cat([x, x_8], dim=1)
            x = self.upsamplex2(self.relu(self.conv_fpn3(x)))
            x = torch.cat([x, x_4], dim=1)
            x = self.upsamplex2(self.relu(self.conv_fpn4(x)))
            x = torch.cat([x, x_2], dim=1)
            x = self.upsamplex2(self.relu(self.conv_fpn5(x)))
        elif self.backbone_stages_num == 4:
            if feats is None:
                x_2, x_4, x_8, x_16 = self._forward_resnet_with_feats(x)
            else:
                x_2, x_4, x_8, x_16 = feats
            # FPN layers
            x = self.upsamplex2(self.relu(self.conv_fpn2(x_16)))
            x = torch.cat([x, x_8], dim=1)
            x = self.upsamplex2(self.relu(self.conv_fpn3(x)))
            x = torch.cat([x, x_4], dim=1)
            x = self.upsamplex2(self.relu(self.conv_fpn4(x)))
            x = torch.cat([x, x_2], dim=1)
            x = self.upsamplex2(self.relu(self.conv_fpn5(x)))
        else:
            raise NotImplementedError(self.resnet_stages_num)
        return x


class BiBackbone(nn.Module):
    """input bitemporal images, with bitemporal feature interactions"""
    def __init__(self, backbone_name='resnet18',
                 backbone_stages_num=5, backbone_stage_start_id=0,
                 pretrained=True,
                 structure='simple5',
                 with_out_conv=True, out_upsample2x=False, output_nc=64,
                 backbone_out_feats=False,
                 bi_inter_type='local_attn', inter_levels=[1,2,3,4], # one which stage to perform interation, [0,1,2,3,4,5]
                 scale_aware=False, scale_levels=(0, 1),
                 window_size=8,):
        super(BiBackbone, self).__init__()
        self.backbone_name = backbone_name
        self.backbone, self.mid_layers = get_backbone(backbone_name, pretrained=pretrained,
                                           backbone_stages_num=backbone_stages_num,
                                           structure=structure, ret_mid_layer_channels=True)
        self.backbone_stages_num = backbone_stages_num
        self.backbone_stage_start_id = backbone_stage_start_id
        self.stages = self._make_stages()
        self.bi_inter_type = bi_inter_type
        self.inter_levels = inter_levels
        self.bi_inter_models = nn.ModuleList()
        self.scale_aware = scale_aware
        if 'local_att' in self.bi_inter_type:
            from models.swin_transformers import Window_att_CD_Block
            import math
            channels_list = [3] + self.mid_layers
            if isinstance(window_size, int):
                window_size = [window_size, ] * len(self.inter_levels)
            for i, level in enumerate(self.inter_levels):  # 0,1,2,3,4,5
                # TODO: add conv to reduce feat dimension
                if level == 0:
                    num_heads = 1
                else:
                    num_heads = 8
                bi_inter = Window_att_CD_Block(dim=channels_list[level],
                                          input_resolution=round(256/(math.pow(2, level))),
                                          window_size=window_size[i], num_heads=num_heads,
                                          with_norm=False)  # 去掉后面的norm部分

                self.bi_inter_models.append(bi_inter)

        elif 'self_att' in self.bi_inter_type:
            from models.transformers import AttentionInterface
            channels_list = [3] + self.mid_layers
            for level in self.inter_levels:
                att = AttentionInterface(dim=channels_list[level])
                self.bi_inter_models.append(att)
        else:
            print('use no bitemporal interactions...')
        self.structure = structure
        self.backbone_out_feats = backbone_out_feats
        if backbone_out_feats:
            assert with_out_conv is False
            self.out_feat_channels = self.mid_layers

        if structure == 'fcn':
            self.single_decoder = FCN_Decoder(channels=self.mid_layers)
            layers = self.mid_layers[0]
        else:
            layers = 0
        self.with_out_conv = with_out_conv
        if self.with_out_conv:
            self.conv_pred = nn.Conv2d(layers, output_nc, kernel_size=3, padding=1)
            self.out_layers = output_nc

        self.save_feat = False
        self.feat_vis_b = [[], []]
        self.feat_vis_a = [[], []]

    def forward(self, x1, x2, scale1=1, scale2=1, return_feat=False):
        if self.scale_aware:
            feats1, feats2 = self.forward_feats_scale(x1, x2, scale1, scale2)
        else:
            feats1, feats2 = self.forward_feats(x1, x2)
        if return_feat or self.backbone_out_feats:
            return feats1, feats2
        if self.structure == 'fcn':
            x1 = self.single_decoder(feats1)
            x2 = self.single_decoder(feats2)
            if self.with_out_conv:
                x1 = self.conv_pred(x1)
                x2 = self.conv_pred(x2)
            return x1, x2
        else:
            return feats1[-1], feats2[-1]

    def forward_feats(self, x1, x2):
        feats_x1 = []
        feats_x2 = []
        level_id = 0
        if self.save_feat:
            self.feat_vis_b = [[], []]
            self.feat_vis_a = [[], []]
        if 0 in self.inter_levels:
            x1, x2 = self.bi_inter_models[level_id](x1, x2)
            level_id += 1
        for i, stage in enumerate(self.stages):
            x1 = stage(x1)
            x2 = stage(x2)
            if i + 1 in self.inter_levels:
                if self.save_feat:
                    self.feat_vis_b[0].append(x1.clone())
                    self.feat_vis_b[1].append(x2.clone())
                x1, x2 = self.bi_inter_models[level_id](x1, x2)
                level_id += 1
                if self.save_feat:
                    self.feat_vis_a[0].append(x1.clone())
                    self.feat_vis_a[1].append(x2.clone())
            feats_x1.append(x1)
            feats_x2.append(x2)
        return feats_x1, feats_x2

    def forward_feats_scale(self, x1, x2, scale1=1, scale2=1):
        """
        Note: x1: low resolution,
            x2: high resolution, as ref
        """
        feats_x1 = []
        feats_x2 = []
        level_id = 0
        if 0 in self.scale_levels:
            x1 = self.bi_inter_models[level_id](x1, x2, scale=scale1, scale2=scale2)
            level_id += 1
        for i, stage in enumerate(self.stages):
            x1 = stage(x1)
            x2 = stage(x2)
            if i + 1 in self.scale_levels:
                x1 = self.bi_inter_models[level_id](x1, x2, scale=scale1, scale2=scale2)
                level_id += 1
            feats_x1.append(x1)
            feats_x2.append(x2)
        return feats_x1, feats_x2

    def _make_stages(self):
        if 'resnet' in self.backbone_name:
            stages = self.backbone.get_stages()
        else:
            raise NotImplementedError
        return stages[self.backbone_stage_start_id:self.backbone_stages_num]


class BaseCD(torch.nn.Module):
    def __init__(self, input_nc, output_nc, output_feat_nc=32,
                 backbone_stages_num=5, backbone_name='resnet18',
                 backbone_out_upsample2x=False,
                 pretrained: Union[bool, str] = True,
                 backbone_structure='simple3',
                 backbone_fuse_mode='add',
                 head_pretrained=False,
                 frozen_backbone_weights=False,
                 fuse_mode='sub', output_sigmoid=False,
                 scale_aware=False, scale_levels=(1,2,3,4,5)):
        super(BaseCD, self).__init__()
        self.backbone = Backbone(input_nc=input_nc, output_nc=output_feat_nc,
                                 backbone_stages_num=backbone_stages_num,
                                 backbone_name=backbone_name,
                                 pretrained=pretrained,
                                 backbone_fuse_mode=backbone_fuse_mode,
                                 structure=backbone_structure,
                                 out_upsample2x=backbone_out_upsample2x,
                                 head_pretrained=head_pretrained,
                                 frozen_backbone_weights=frozen_backbone_weights,
                                 scale_aware=scale_aware, scale_levels=scale_levels)
        self.fuse_mode = fuse_mode
        if self.fuse_mode is 'sub':
            self.classifier = TwoLayerConv2d(in_channels=output_feat_nc, out_channels=output_nc)
        elif self.fuse_mode is 'cat':
            self.classifier = TwoLayerConv2d(in_channels=output_feat_nc*2, out_channels=output_nc)
        else:
            raise NotImplementedError
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2=None, is_train=False):
        if x2 is None:
            x2 = x1
        b, c, h, w = x1.shape
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        if self.fuse_mode is 'sub':
            x = torch.abs(x1 - x2)
        elif self.fuse_mode is 'cat':
            x = torch.cat([x1, x2], dim=1)
        else:
            raise NotImplementedError
        x = self.classifier(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear',
                          align_corners=False)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x


class CD_INR(nn.Module):
    """
    Change detection with implicit feature alignement
    """
    def __init__(self, input_nc, output_nc=2, output_feat_nc=32,
                 backbone_stages_num=5, backbone_name='resnet18', backbone_stage_start_id=0,
                 backbone_out_upsample2x=False,
                 pretrained: Union[bool, str] = True,
                 backbone_structure='simple',
                 backbone_inter=False, bi_inter_type='bit', inter_levels=[2,],
                 window_size=8,
                 head_pretrained=False,
                 frozen_backbone_weights=False,
                 backbone_out_feats=True,
                 level_n=4,
                 fuse_mode='concat', ex_temporal_random=False,
                 learn_pe=False,
                 out_size=(64, 64),  # size of output feat from ifa
                 with_edge_info=False, with_edge_type='sobel', edge_mode='simple', edge_dim=3,
                 ):
        super(CD_INR, self).__init__()
        self.backbone_inter = backbone_inter
        if backbone_inter is False:
            self.backbone = Backbone(input_nc=input_nc, output_nc=output_feat_nc,
                                 backbone_stages_num=backbone_stages_num,
                                 backbone_name=backbone_name,
                                 pretrained=pretrained,
                                 structure=backbone_structure,
                                 out_upsample2x=backbone_out_upsample2x,
                                 head_pretrained=head_pretrained,
                                 frozen_backbone_weights=frozen_backbone_weights,
                                 backbone_out_feats=backbone_out_feats,
                                 with_out_conv=False)
        else:
            self.backbone = BiBackbone(backbone_name=backbone_name,
                                       backbone_stages_num=backbone_stages_num,
                                       backbone_stage_start_id=backbone_stage_start_id,
                                       pretrained=pretrained,
                                       structure=backbone_structure,
                                       output_nc=output_feat_nc,
                                       scale_levels=inter_levels,
                                       bi_inter_type=bi_inter_type,
                                       inter_levels=inter_levels, window_size=window_size,
                                       backbone_out_feats=backbone_out_feats,
                                       with_out_conv=False)
        self.valid_stages_num = backbone_stages_num - backbone_stage_start_id
        feat_channels = self.backbone.out_feat_channels
        self.mid_convs = nn.ModuleList()
        self.level_n = level_n
        for i in range(level_n):
            conv = nn.Sequential(nn.Conv2d(feat_channels[5 - self.level_n + i], output_feat_nc, (1, 1)))
            self.mid_convs.append(conv)

        self.fuse_mode = fuse_mode
        if self.fuse_mode == 'sub':
            channels = output_feat_nc
        else:
            channels = output_feat_nc * 2
        self.with_edge_info = with_edge_info
        if self.with_edge_info:
            self.edger = EdgeBtNeck(out_size=out_size, with_edge_type=with_edge_type,
                                    edge_dim=edge_dim, mode=edge_mode)
            additional_in_dim = edge_dim
        else:
            additional_in_dim = 0
        self.ifa_fuse = IFA_MultiLevel(level_n=level_n, in_channels=channels,
                                       out_channels=channels, learn_pe=learn_pe,
                                       size=out_size, additional_in_dim=additional_in_dim)
        self.ex_temporal_random = ex_temporal_random
        self.classifier = TwoLayerConv2d(in_channels=channels, out_channels=output_nc)
        # for feature visualization
        self.save_feat = True
        self.vis = []

    def forward(self, x1, x2=None, is_train=False, info=None):
        if x2 is None:
            x2 = x1
        b, c, h, w = x1.shape
        #  encoder
        if self.backbone_inter:
            feats1, feats2 = self.backbone(x1, x2)
        else:
            feats1 = self.backbone(x1)
            feats2 = self.backbone(x2)
        feats1 = feats1[self.valid_stages_num - self.level_n:]
        feats2 = feats2[self.valid_stages_num - self.level_n:]
        if self.save_feat:
            self.vis = []
            self.vis.append(feats1)
            self.vis.append(feats2)
            # self.vis = self.backbone.feat_vis_b,self.backbone.feat_vis_a
        # middle conv
        feats1 = [conv(feat) for feat, conv in zip(feats1, self.mid_convs)]
        feats2 = [conv(feat) for feat, conv in zip(feats2, self.mid_convs)]

        if self.fuse_mode == 'sub':
            feats = [torch.abs(feat1-feat2) for feat1, feat2 in zip(feats1, feats2)]
        elif self.fuse_mode == 'concat':
            if self.ex_temporal_random:
                if torch.rand(1).item() > 0.5:
                    feats1, feats2 = feats2, feats1
            feats = [torch.concat([feat1, feat2], dim=1) for feat1, feat2 in zip(feats1, feats2)]
        else:
            raise NotImplementedError
        # ifa fuse
        if self.with_edge_info:
            x_a = self.edger(x1, x2)
        else:
            x_a = None
        x = self.ifa_fuse(feats, x_a)
        x = self.classifier(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x



class IFA_MultiLevel(nn.Module):
    """ IFA 隐式多尺度特征对齐，并输出fuse后的特征
    IFA是LIIF方法的扩展，除了相对格点的xy偏移外，还可以包括额外的PE项；
    option：
        unfold：整合局部3*3的特征；
        local：
    """
    def __init__(self, in_channels=64, out_channels=64,  level_n=4,
                 size=None, additional_in_dim=0, with_area_info=True,
                 ultra_pe=False, pos_dim=24, local=False, unfold=False, stride=1,
                 learn_pe=False, require_grad=False,
                 mlp_layers_n=1, out_rgb=False,
                 additional_pe=False,
                 ):
        super(IFA_MultiLevel, self).__init__()
        self.pos_dim = pos_dim
        self.ultra_pe = ultra_pe
        self.learn_pe = learn_pe

        self.local = local
        self.unfold = unfold
        self.stride = stride
        self.size = size
        self.additional_in_dim = additional_in_dim  # 额外的细节信息
        self.with_area_info = with_area_info
        if learn_pe:
            self.pes = nn.ModuleList()
            for i in range(level_n):
                # TODO: embedding to be shared
                self.pes.append(PositionEmbeddingLearned(self.pos_dim//2))
                print(f'learned pe...')
        elif ultra_pe:
            self.pes = nn.ModuleList()
            for i in range(level_n):
                self.pes.append(SpatialEncoding(2, self.pos_dim, require_grad=require_grad))
            self.pos_dim += 2
        else:
            self.pos_dim = 2
        in_dim = level_n*(in_channels + self.pos_dim)
        if unfold:
            in_dim = level_n*(in_channels*9 + self.pos_dim)
        if with_area_info:
            in_dim += level_n*2
        in_dim += self.additional_in_dim

        if mlp_layers_n == 1:
            self.imnet = nn.Sequential(
                nn.Conv1d(in_dim, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU())
        elif mlp_layers_n == 2:
            self.imnet = nn.Sequential(
                nn.Conv1d(in_dim, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU(),)
        elif mlp_layers_n == 3:
            self.imnet = nn.Sequential(
                nn.Conv1d(in_dim, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU(),)
        elif mlp_layers_n == 4:
            self.imnet = nn.Sequential(
                nn.Conv1d(in_dim, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU(),)
        else:
            raise NotImplementedError
        self.out_rgb = out_rgb  # 若输出rgb则再映射一波
        if out_rgb:
            self.out_mlp = nn.Conv1d(out_channels, 3, 1)

    def forward(self, xs, x_a=None):
        h, w = self.size if self.size is not None else xs[0].shape[2:]
        bs = xs[0].shape[0]
        context = []
        for i, feat in enumerate(xs):
            context.append(self.forward_single_level(feat, size=[h, w], level=i))
        context = torch.cat(context, dim=-1).permute(0, 2, 1)  # bs, c, (h*w)
        if x_a is not None:
            # print(x_a.shape,'.......................')
            if x_a.shape[-1] != w:
                x_a = F.interpolate(x_a, size=[h, w])
            x_a = x_a.view(bs, -1, h*w)
            context = torch.cat([context, x_a], dim=1)
        out = self.imnet(context)
        if self.out_rgb:
            out = self.out_mlp(out)
        return out.view(bs, -1, h, w)

    def forward_single_level(self, x, size, level=0):
        h, w = size
        if not self.local:
            if self.unfold:
                x = F.unfold(x, 3, padding=1).view(x.shape[0],x.shape[1]*9, x.shape[2], x.shape[3])
            rel_coord, q_feat = ifa_feat(x, [h, w])
            rel_coord_ = rel_coord
            if self.ultra_pe or self.learn_pe:
                rel_coord = self.pes[level](rel_coord)
            x = torch.cat([rel_coord, q_feat], dim=-1)
            if self.with_area_info:
                scale_info = torch.ones_like(rel_coord_)
                scale_info[:, :, 0] = 2 / 256 * h  # TODO: 256 is 原图大小
                scale_info[:, :, 1] = 2 / 256 * w
                x = torch.cat([x, scale_info], dim=-1)
        else:
            if self.unfold:
                x = F.unfold(x, 3, padding=1).view(x.shape[0],x.shape[1]*9, x.shape[2], x.shape[3])
            rel_coord_list, q_feat_list, area_list = ifa_feat(x, [h, w],  local=True, stride=self.stride)
            total_area = torch.stack(area_list).sum(dim=0)
            context_list = []
            for rel_coord, q_feat, area in zip(rel_coord_list, q_feat_list, area_list):
                if self.ultra_pe or self.learn_pe:
                    rel_coord = self.pes[level](rel_coord)
                    # rel_coord = eval('self.pos'+str(level))(rel_coord)
                cont = torch.cat([rel_coord, q_feat], dim=-1)
                if self.with_area_info:
                    scale_info = torch.ones_like(rel_coord)
                    scale_info[:,:, 0] = 2 / 256 * h  # TODO: 256 is 原图大小
                    scale_info[:,:, 1] = 2 / 256 * w
                    cont = torch.cat([cont, scale_info], dim=-1)
                context_list.append(cont)
            ret = 0
            t = area_list[0]; area_list[0] = area_list[3]; area_list[3] = t
            t = area_list[1]; area_list[1] = area_list[2]; area_list[2] = t
            for conte, area in zip(context_list, area_list):
                x = ret + conte * ((area / total_area).unsqueeze(-1))
        # x = x.view(x.shape[0], -1, h, w)
        return x


class SpatialEncoding(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 sigma=6,
                 cat_input=True,
                 require_grad=False, ):

        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"

        n = out_dim // 2 // in_dim
        m = 2 ** np.linspace(0, sigma, n)
        m = np.stack([m] + [np.zeros_like(m)] * (in_dim - 1), axis=-1)
        m = np.concatenate([np.roll(m, i, axis=-1) for i in range(in_dim)], axis=0)
        self.emb = torch.FloatTensor(m)
        if require_grad:
            self.emb = nn.Parameter(self.emb, requires_grad=True)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.cat_input = cat_input
        self.require_grad = require_grad

    def forward(self, x):

        if not self.require_grad:
            self.emb = self.emb.to(x.device)
        y = torch.matmul(x, self.emb.T)
        if self.cat_input:
            return torch.cat([x, torch.sin(y), torch.cos(y)], dim=-1)
        else:
            return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def ifa_feat(res, size, stride=1, local=False):
    bs, hh, ww = res.shape[0], res.shape[-2], res.shape[-1]
    h, w = size
    coords = (make_coord((h, w)).cuda().flip(-1) + 1) / 2
    # coords = (make_coord((h,w)).flip(-1) + 1) / 2
    coords = coords.unsqueeze(0).expand(bs, *coords.shape)
    coords = (coords * 2 - 1).flip(-1)

    feat_coords = make_coord((hh, ww), flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(res.shape[0], 2,
                                                                                                  *(hh, ww))
    # feat_coords = make_coord((hh,ww), flatten=False).permute(2, 0, 1) .unsqueeze(0).expand(res.shape[0], 2, *(hh,ww))

    if local:
        vx_list = [-1, 1]
        vy_list = [-1, 1]
        eps_shift = 1e-6
        rel_coord_list = []
        q_feat_list = []
        area_list = []
    else:
        vx_list, vy_list, eps_shift = [0], [0], 0
    rx = stride / h
    ry = stride / w

    for vx in vx_list:
        for vy in vy_list:
            coords_ = coords.clone()
            coords_[:, :, 0] += vx * rx + eps_shift
            coords_[:, :, 1] += vy * ry + eps_shift
            coords_.clamp_(-1 + 1e-6, 1 - 1e-6)
            q_feat = F.grid_sample(res, coords_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)
            q_coord = F.grid_sample(feat_coords, coords_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:,
                      :, 0, :].permute(0, 2, 1)
            rel_coord = coords - q_coord
            rel_coord[:, :, 0] *= hh  # res.shape[-2]
            rel_coord[:, :, 1] *= ww  # res.shape[-1]
            if local:
                rel_coord_list.append(rel_coord)
                q_feat_list.append(q_feat)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                area_list.append(area + 1e-9)

    if not local:
        return rel_coord, q_feat  # b, n, c
    else:
        return rel_coord_list, q_feat_list, area_list


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.row_embed = nn.Embedding(256, num_pos_feats)
        self.col_embed = nn.Embedding(256, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        # input: x, [b, N, 2]
        # output: [b, N, C]

        h = w = int(np.sqrt(x.shape[1]))
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1).view(x.shape[0], h * w, -1)
        # print('pos', pos.shape)
        return pos


#
if __name__ == '__main__':
    device = torch.device('cuda:0')
    pretrained = r'G:\program\CD\CD4_1\checkpoints\SSL_simsiam_resnet18_imagenet_LEVIR_b64_lr0.001_train_val_200_linear_sgd\pretrained.pth'
    # model = Backbone(backbone_name='resnet18', input_nc=3, output_nc=64,
    #                       structure='fcn', pretrained=pretrained,
    #                  head_pretrained=False).to(device)
    model = BaseCD(input_nc=3, output_nc=2, backbone_name='resnet18',
                 output_feat_nc=64,
                 pretrained=False,
                 backbone_out_upsample2x=True).to(device)

    model = CD_INR(input_nc=3, backbone_name='resnet18', output_feat_nc=64,
                 pretrained=pretrained, frozen_backbone_weights=False,
                 fuse_mode='concat', ex_temporal_random=False, out_size=(128, 128),
                 with_edge_info=True, with_edge_type='canny',
                 edge_mode='filter_conv', edge_dim=3, learn_pe=True,
                 ).to(device)
    data_in = torch.randn([2, 3, 256, 256], dtype=torch.float32).to(device)
    init_weights(model)
    data_out = model(data_in)

    print(type(data_out))
    if isinstance(data_out, tuple):
        print('shape of the output: ', data_out[0].shape)
    else:
        print('shape of the output: ', data_out.shape)
