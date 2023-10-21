import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet18


def build_backbone(backbone, pretrained=False, strides=[2,2,2,2,2]):
    if backbone == 'resnet18':
        return resnet18(pretrained=pretrained, strides=strides)
    else:
        raise NotImplementedError


def build_decoder(fc, BatchNorm):
    return Decoder(fc, BatchNorm)


class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, fc, BatchNorm=nn.BatchNorm2d):
        super(Decoder, self).__init__()
        self.fc = fc
        self.dr2 = DR(64, 96)
        self.dr3 = DR(128, 96)
        self.dr4 = DR(256, 96)
        self.dr5 = DR(512, 96)
        self.last_conv = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
                                       BatchNorm(self.fc),
                                       nn.ReLU(),
                                       )
        self._init_weight()

    def forward(self, x, low_level_feat2, low_level_feat3, low_level_feat4):
        x2 = self.dr2(low_level_feat2)
        x3 = self.dr3(low_level_feat3)
        x4 = self.dr4(low_level_feat4)
        x = self.dr5(x)

        x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x2.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, x2, x3, x4), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class CDNet(nn.Module):
    def __init__(self, backbone_name='resnet18', output_stride=16, f_c=64, freeze_bn=False, in_c=3):
        super(CDNet, self).__init__()
        BatchNorm = nn.BatchNorm2d

        # self.transform = get_transform(convert=True, normalize=True)

        self.backbone = build_backbone(backbone_name, strides=[2,2,2,2,1])
        self.decoder = build_decoder(f_c, BatchNorm)

        self.cbam0 = CBAM(64)
        self.cbam1 = CBAM(64)

        self.cbam2 = CBAM(64)
        self.cbam3 = CBAM(128)
        self.cbam4 = CBAM(256)
        self.cbam5 = CBAM(512)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, hr_img1, hr_img2=None, is_train=False):
        if hr_img2 is None:
            hr_img2 = hr_img1
        f1_d = self.backbone.forward_feats(hr_img1)
        f2_d = self.backbone.forward_feats(hr_img2)
        x_1, f2_1, f3_1, f4_1 = f1_d['p5'], f1_d['p2'], f1_d['p3'], f1_d['p4']
        x_2, f2_2, f3_2, f4_2 = f2_d['p5'], f2_d['p2'], f2_d['p3'], f2_d['p4']

        x1 = self.decoder(self.cbam5(x_1), self.cbam2(f2_1), self.cbam3(f3_1), self.cbam4(f4_1))
        x2 = self.decoder(self.cbam5(x_2), self.cbam2(f2_2), self.cbam3(f3_2), self.cbam4(f4_2))

        x1 = self.cbam0(x1)
        x2 = self.cbam0(x2)
        import einops
        x1 = einops.rearrange(x1, 'b c h w->b h w c')
        x2 = einops.rearrange(x2, 'b c h w->b h w c')
        dist = F.pairwise_distance(x1, x2, keepdim=True)  # TODO, 更新后的torch，在最后一维度做dis
        dist = einops.rearrange(dist, 'b h w c->b c h w')
        dist = F.interpolate(dist, size=hr_img1.shape[2:], mode='bicubic', align_corners=True)

        return dist

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()