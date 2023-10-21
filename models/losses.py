import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def define_loss(loss_name):
    if loss_name == 'ce':
        loss_func = cross_entropy
    elif loss_name == 'bce':
        loss_func = binary_ce
    elif loss_name == 'DTCDSN_loss':
        loss_func = DTCDSN_loss()
    elif loss_name == 'deep_ce_dice':  # out dim 1
        loss_func = deep_ce_dice_loss
    elif loss_name == 'deep_ce':
        loss_func = deep_ce_loss
    elif loss_name == 'bcl':
        loss_func = BCL(margin=2)
    elif loss_name == 'ce_iou':
        loss_func = LossCEIOU()
    elif loss_name == 'focal_dice':
        loss_func = focal_dice_loss
    else:
        raise NotImplemented(loss_name)

    return loss_func


class BalancedCrossEntropyLoss(torch.nn.Module):
    """
    from https://github.com/facebookresearch/astmt/blob/master/fblib/layers/loss.py
    and
    # https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation/blob/main/pretrain/modules/losses.py
    Balanced Cross Entropy Loss with optional ignore regions
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, preds, labels, void_pixels=None):
        assert (preds.size() == labels.size())

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            #  label：正负两类
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(preds, 0).float()
        loss_val = torch.mul(preds, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(preds - 2 * torch.mul(preds, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(labels.size()))
        elif self.batch_average:
            final_loss /= labels.size()[0]

        return final_loss


class BinaryCrossEntropyLoss(torch.nn.Module):
    """
    from https://github.com/facebookresearch/astmt/blob/master/fblib/layers/loss.py
    Binary Cross Entropy with ignore regions, not balanced.
    """

    def __init__(self, size_average=True, batch_average=True):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())

        labels = torch.ge(label, 0.5).float()

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)
        final_loss = loss_pos + loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=-100):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if target.dim() == 3:
        if input.shape[-1] != target.shape[-1]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


def binary_ce(input, target, weight=None, reduction='none', ignore_index=255):
    assert input.shape[1] == 1
    B, C, H, W = input.shape
    target = target.float()

    input = input.reshape([-1])
    target = target.reshape([-1])

    loss = F.binary_cross_entropy_with_logits(input, target, weight=weight,
                                                        reduction=reduction)
    num = (target != ignore_index).float().sum().detach()
    # num = B * C * H * W
    loss = loss.sum() / num
    return loss


class DTCDSN_loss(nn.Module):
    """

    from DTCDSN model
    """
    def __init__(self, gamma = 1.5, size_average=True):
        super(DTCDSN_loss,self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, prob, target):
        target = target.view(-1)
        #prob = F.sigmoid(logit)
        prob = prob.view(-1)
        prob_p = torch.clamp(prob, 1e-8, 1 - 1e-8)
        prob_n = torch.clamp(1.0 - prob, 1e-8, 1 - 1e-8)
        batch_loss= - torch.pow((2 - prob_p),self.gamma) * prob_p.log() * target \
                    - prob_n.log() * (1 - target) *(2 - prob_n)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss


def deep_ce_dice_loss(inputs, target):
    target = target.float()
    losses = 0
    for input in inputs:
        h, w = input.shape[2], input.shape[3]
        target_ = torch.nn.functional.adaptive_max_pool2d(target, [h, w])

        losses = losses + ce_dice_loss(input, target_)
    return losses


def deep_ce_loss(inputs, target):
    assert isinstance(inputs, tuple)
    target = target.float()
    losses = 0
    for input in inputs:
        h, w = input.shape[2], input.shape[3]
        target_ = torch.nn.functional.adaptive_max_pool2d(target, [h, w])

        losses = losses + cross_entropy(input, target_)
    return losses


def ce_dice_loss(input, target):
    if target.ndim == 3:
        target = target.unsqueeze(1)
    assert target.ndim == 4

    bce_loss = nn.BCELoss()
    bce_loss = bce_loss(torch.sigmoid(input), target)
    dic_loss = dice_loss(input, target.long())
    loss = bce_loss + dic_loss
    return loss


def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1).cpu()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    true_1_hot = true_1_hot.to(logits.device)
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """
    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label == 1] = -1
        label[label == 0] = 1

        mask = (label != 255).float()
        distance = distance * mask

        pos_num = torch.sum((label==1).float())+0.0001
        neg_num = torch.sum((label==-1).float())+0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) /pos_num  # no-change
        loss_2 = torch.sum((1-label) / 2 *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        ) / neg_num    # change
        loss = loss_1 + loss_2
        return loss

class LossCEIOU(nn.Module):
    def __init__(self, weight_ba_loss=0.67, weight_ce_loss=0.33):
        super(LossCEIOU, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=1)
        torch.nn.init.constant(self.bn.weight, 1)
        self.bn.cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 36])).cuda()
        self.weight_ba_loss = weight_ba_loss
        self.weight_ce_loss = weight_ce_loss

    def forward(self, y, lbl):
        lbl = lbl.squeeze(1)
        ce_loss = self.ce_loss(y, lbl)
        diff = y[:, 1] - y[:, 0]  # 第1维大的为changed
        diff = torch.unsqueeze(diff, 1)
        diff = self.bn(diff)
        diff = torch.sigmoid(diff)
        lbl_float = lbl.float()
        iou_loss = 1-torch.sum(diff * lbl_float) / torch.sum(diff + lbl_float - diff * lbl_float)
        loss = iou_loss * self.weight_ba_loss+ce_loss*self.weight_ce_loss
        return loss


def focal_dice_loss(predictions, target):
    """Calculating the loss"""
    loss = 0
    if not isinstance(predictions, tuple) and not isinstance(predictions, list):
        predictions = (predictions, )
    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)
    for prediction in predictions:
        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss

from torch.autograd import Variable
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))


        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


if __name__ == '__main__':
    # input is of size N x C = 3 x 5

    input = torch.randn(3, 3, 16, 16, requires_grad=True)
    # each element in target has to have 0 <= value < C
    target = torch.randn(3, 3, 16, 16, requires_grad=True) > 0.5
    target = target.long()

    Loss = BalancedCrossEntropyLoss()
    loss = Loss(input, target)
    print('input： %f\n', input)
    print('target： %f\n', target)
    print('loss： %f\n', loss)
