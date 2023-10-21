from einops import rearrange
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import models
import math


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def positionalencoding_cd(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: 2 * d_model*height*width position matrix for two temporal matrix
    """
    if (d_model-2) % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))

    pe = torch.zeros(d_model, height, width, 2)
    # Each dimension use half of d_model
    d_model = int((d_model-2) / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).unsqueeze(-1).repeat(1, height, 1, 2)
    pe[1:d_model:2, :, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).unsqueeze(-1).repeat(1, height, 1, 2)
    pe[d_model:d_model * 2:2, :, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).unsqueeze(-1).repeat(1, 1, width,2)
    pe[d_model + 1:d_model * 2+1:2, :, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).unsqueeze(-1).repeat(1, 1, width,2)

    pe[d_model * 2, :, :, 0] = 1
    pe[d_model * 2 + 1, :, :, 0] = 0
    pe[d_model * 2, :, :, 1] = 0
    pe[d_model * 2 + 1, :, :, 1] = 1

    return pe

def positionalencoding_cd_1t(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: 2 * d_model*height*width position matrix for two temporal matrix
    """
    if (d_model-2) % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))

    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int((d_model-2) / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model:d_model * 2:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1:d_model * 2+1:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    pe[d_model * 2, :, :] = 0.5
    pe[d_model * 2 + 1, :, :] = 0.5

    return pe

class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # vis_tmp2(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, y_dim, x_dim = input_tensor.size()

        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2).type_as(input_tensor)
        yy_channel = yy_channel.float() / y_dim
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        ret = torch.cat([input_tensor, yy_channel], dim=1)

        return ret


class CoordConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        in_size = in_channels + 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = AddCoords()(x)
        ret = self.conv(ret)
        return ret


class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True,
                 fuse_mode='sub',
                 pretrained='imagenet',
                 structure='simple3'):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        self.fuse_mode = fuse_mode
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
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.structure = structure
        self.backbone_stages_num = resnet_stages_num
        self.resnet_stages_num = resnet_stages_num
        if self.backbone_stages_num == 5:
            layers = 512 * expand
        elif self.backbone_stages_num == 4:
            layers = 256 * expand
        elif self.backbone_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        if self.structure == 'fcn':
            assert 'resnet' in backbone
            self.expand = 1
            if backbone == 'resnet50':
                self.expand = 4
            if self.backbone_stages_num == 5:
                self.conv_fpn1 = nn.Conv2d(512 * self.expand,
                                           256 * self.expand, kernel_size=3, padding=1)
            if self.backbone_stages_num >= 4:
                self.conv_fpn2 = nn.Conv2d(256 * self.expand, 128 * self.expand, kernel_size=3, padding=1)

            self.conv_fpn3 = nn.Conv2d(128 * self.expand, 64 * self.expand, kernel_size=3, padding=1)
            layers = 64 * self.expand
        elif self.structure == 'unet':
            assert 'resnet' in backbone
            self.expand = 1
            if backbone == 'resnet50':
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
            layers = 32 * self.expand
        elif 'simple' not in self.structure:
            raise NotImplementedError

        if self.fuse_mode is 'sub':
            self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        elif self.fuse_mode is 'cat':
            self.classifier = TwoLayerConv2d(in_channels=64, out_channels=output_nc)
        else:
            raise NotImplementedError

        self.if_upsample_2x = if_upsample_2x
        self.upsample_last = True

        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2=None):
        """
        In the forward function we accept two Tensors of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if x2 is None:
            x2 = x1
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        if self.fuse_mode is 'sub':
            x = torch.abs(x1 - x2)
        elif self.fuse_mode is 'cat':
            x = torch.cat([x1, x2], dim=1)
        else:
            raise NotImplementedError
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        if self.upsample_last:
            x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if 'simple' in self.structure:
            x = self.forward_resnet(x)
        elif self.structure == 'fcn':
            x = self.forward_resnet_fcn(x)
        elif self.structure == 'unet':
            x = self.forward_resnet_unet(x)
        if self.if_upsample_2x:
            x = self.upsamplex2(x)
        # output layers
        x = self.conv_pred(x)
        if not self.upsample_last:
            x = self.upsamplex4(x)
        return x

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

    def forward_resnet_fcn(self, x):
        if self.backbone_stages_num == 5:
            x_4, x_8, x_16, x_32 = self._forward_resnet_with_feats(x)[1:]
            # FPN layers
            x = self.upsamplex2(self.relu(self.conv_fpn1(x_32)))
            x = self.upsamplex2(self.relu(self.conv_fpn2(x + x_16)))
            x = self.upsamplex2(self.relu(self.conv_fpn3(x + x_8)))
        elif self.backbone_stages_num == 4:
            x_4, x_8, x_16 = self._forward_resnet_with_feats(x)[1:]
            # FPN layers
            x = self.upsamplex2(self.relu(self.conv_fpn2(x_16)))
            x = self.upsamplex2(self.relu(self.conv_fpn3(x + x_8)))
        else:
            raise NotImplementedError(self.resnet_stages_num)
        return x

    def forward_resnet_unet(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if self.backbone_stages_num == 5:
            x_2, x_4, x_8, x_16, x_32 = self._forward_resnet_with_feats(x)
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
            x_2, x_4, x_8, x_16 = self._forward_resnet_with_feats(x)
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


class BASE_Transformer(ResNet):
    """resnet 8 downsampling"""
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, recurrent_num=1,  token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True, pool_mode='max', pool_size=2,
                 with_token_m=False, backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True,
                 fuse_mode='sub',
                 pretrained='imagenet', structure='simple3'):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               fuse_mode=fuse_mode,
                                               pretrained=pretrained,
                                               structure=structure)

        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  如果不使用tokenzier，则使用pool的方式，把特征图降采样到每个尺度
            # pool_size = 32
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        self.recurrent_num = recurrent_num
        dim = 32
        mlp_dim = 2*dim
        self.with_token_m = with_token_m
        if self.with_token_m:
            self.tokens_memory = nn.Parameter(torch.zeros(1, self.token_len, 32),
                                              requires_grad=False)
            self.m = 0.99  # momentum update

        self.with_pos = with_pos
        if with_pos is 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        # elif with_pos is 'fix':
        #     from models.position_encoding import PositionEmbeddingSine
        #     self.pos_embedding =
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        elif self.with_decoder_pos == 'fix':
            from models.position_encoding import PositionEmbeddingSine
            x = torch.randn([1,decoder_pos_size,decoder_pos_size]).cuda()
            self.pos_embedding_decoder = PositionEmbeddingSine(num_pos_feats=16,
                                                               normalize=True)(x)

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        # for visualize images
        self.visualize = False
        self.vis = []
        self.save_features = False

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        if self.visualize:
            att_map = spatial_attention.view(b, self.token_len, h, -1)
            size = att_map.shape[-1]
            # att_map = F.interpolate(att_map, [256, 256], mode='bilinear')
            att_map = F.interpolate(att_map, [size*4, size*4], mode='bicubic')

            self.vis.append(att_map)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    @torch.no_grad()
    def _momentum_update_token_memory(self, token1, token2):
        """
        Momentum update of the token_memory
        """
        #  在b维度叠加
        token = torch.cat([token1, token2], dim=0)
        mean_token = torch.mean(token, dim=0, keepdim=True)
        self.tokens_memory.data = self.tokens_memory.data * self.m + mean_token * (1. - self.m)

    def get_grad(self, name=None):
        """返回梯度"""
        shape = self.tokens_.shape
        target = torch.zeros(*shape, device='cuda:0')
        # target[:,0]
        print(target.shape)
        # self.tokens_.retain_grad()
        #  一种方式
        self.tokens.backward(target)
        # 另一种方式
        # t = self.tokens.mean()
        # t.backward(retain_graph=True)

        grad = self.tokens_.grad
        # print(grad)
        # 梯度的channel-wise平均
        g = torch.mean(grad, dim=[-1])
        print('feat-channel-wise', g)

        return g

    def _get_grad_x1(self, x1_):
        x1_ = x1_.requires_grad_(True)
        x1_.retain_grad()

        x1 = self.forward_single(x1_)
        shape = x1.shape
        target = torch.zeros(*shape, device='cuda:0')

        x1.backward(target)
        grad = x1_.grad
        # print(grad)
        # 梯度的channel-wise平均
        g = torch.mean(grad, dim=[-2,-1])
        print('feat-channel-wise', g)

    def forward(self, x1, x2=None, is_train=False):
        if x2 is None:
            x2 = x1
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        if self.save_features:
            self.f1 = x1
            self.f2 = x2
        # self.recurrent_num=0
        for i in range(self.recurrent_num):
            if self.tokenizer:
                token1 = self._forward_semantic_tokens(x1)
                token2 = self._forward_semantic_tokens(x2)
            else:
                token1 = self._forward_reshape_tokens(x1)
                token2 = self._forward_reshape_tokens(x2)
            if self.token_trans:
                self.tokens_ = torch.cat([token1, token2], dim=1)
                # self.tokens_.retain_grad()
                self.tokens = self._forward_transformer(self.tokens_)
                # self.get_grad()
                token1, token2 = self.tokens.chunk(2, dim=1)

            if self.with_token_m:
                token1_ = self.tokens_memory * self.m + token1 * (1. - self.m)
                token2_ = self.tokens_memory * self.m + token2 * (1. - self.m)
                # update the parameter
                self._momentum_update_token_memory(token1, token2)
                token1 = token1_
                token2 = token2_
            if self.with_decoder:
                x1 = self._forward_transformer_decoder(x1, token1)
                x2 = self._forward_transformer_decoder(x2, token2)
            else:
                x1 = self._forward_simple_decoder(x1, token1)
                x2 = self._forward_simple_decoder(x2, token2)

        if self.fuse_mode is 'sub':
            x = torch.abs(x1 - x2)
        elif self.fuse_mode is 'cat':
            x = torch.cat([x1, x2], dim=1)
        else:
            raise NotImplementedError
        if self.save_features:
            self.f1_new = x1
            self.f2_new = x2
            self.f_dif = x
        if not self.if_upsample_2x and 'simple' in self.structure:
            x = self.upsamplex2(x)
        if self.upsample_last:
            x = self.upsamplex4(x)

        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        if self.save_features:
            self.pred = x
        return x

