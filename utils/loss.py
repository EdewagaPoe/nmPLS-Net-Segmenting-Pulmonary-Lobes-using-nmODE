import torch
import torch.nn as nn
import numpy as np

from skimage.segmentation import find_boundaries
from skimage.morphology import closing,binary_dilation
from scipy import ndimage as ndi
# stat = {'1' : rm,
#         '2' : ru,
#         '3' : rl,
#         '4' : lu,
#         '5' : ll,}
# r1:00 11 22 33 44 55
# r2:13 14 15 23 24 25 35
# r3:01 02 03 04 05 
# r4:12 34 45
from torch import Tensor
r1 = [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]]
r2 = [[1,3],[1,4],[1,5],[2,3],[2,4],[2,5],[3,5]]
r3 = [[0,1],[0,2],[0,3],[0,4],[0,5]]
r4 = [[1,2],[3,4],[4,5]]
rs = [r1,r2,r3,r4]

  
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]
class Multi_loss(nn.Module):
    
    def __init__(self,a = 0.5,b = 0.5,lossa=nn.CrossEntropyLoss(),lossb=nn.CrossEntropyLoss()):
        super(Multi_loss, self).__init__()
        
        self.a = a
        self.b = b
        self.lossa = lossa
        self.lossb = lossb
    
    def	forward(self, inputs,target):
        
        loss1 = self.lossa(inputs,target)
        loss2 = self.lossb(inputs,target)
        
        return self.a*loss1 + self.b*loss2
class Multi_loss1(nn.Module):
    
    def __init__(self,a = 0.8,b = 0.2,c=0.1):
        super(Multi_loss1, self).__init__()
        
        self.a = a
        self.b = b
        self.c = c
        self.focal_loss = Focal_loss_seg(gamma = 2)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def	forward(self, inputs1,target,inputs2,border,inputs3,inputs4):
        
        loss1 = self.ce_loss(inputs1,target)
        loss2 = self.focal_loss(inputs2,border)
        loss3 = self.ce_loss(inputs3,target)+self.ce_loss(inputs4,target)
        return self.a*loss1 + self.b*loss2+self.c*loss3
class Multi_loss_ce_dice(nn.Module):
    
    def __init__(self,a = 0.2,b = 0.8):
        super(Multi_loss_ce_dice, self).__init__()
        
        self.a = a
        self.b = b
        self.dice_loss = Multi_label_DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def	forward(self, inputs,target):
        
        loss1 = self.ce_loss(inputs,target)
        loss2 = self.dice_loss(inputs,target)
        
        return self.a*loss1 + self.b*loss2
    
    
class Multi_loss_dice_focal(nn.Module):
    
    def __init__(self,a = 0.5,b = 0.5):
        super(Multi_loss_dice_focal, self).__init__()
        
        self.a = a
        self.b = b
        self.dice_loss = Multi_label_DiceLoss()
        self.focal_loss = Focal_loss_seg(gamma = 2)
    
    def	forward(self, inputs1,target,inputs2,border):
        
        loss1 = self.dice_loss(inputs1,target)
        loss2 = self.focal_loss(inputs2,border)
        
        return self.a*loss1 + self.b*loss2
    

        
        
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def	forward(self, inputs, target):
        N = target.size(0)
        smooth = 1
 
        input_flat = inputs.view(N, -1)
        target_flat = target.view(N, -1)
 
        #intersection = input_flat * target_flat
        intersection = torch.mul(input_flat,target_flat)
 
        num = 2 * intersection.sum() + smooth
        den = input_flat.sum() + target_flat.sum() + smooth
        loss = 1 - num / den
        
        #print(num.item(),den.item(),loss.item())
 
        return loss

class DiceLoss_pow(nn.Module):
    def __init__(self):
        super(DiceLoss_pow, self).__init__()
 
    def	forward(self, inputs, target):
        N = target.size(0)
        smooth = 1
 
        input_flat = inputs.view(N, -1)
        target_flat = target.view(N, -1)
 
        #intersection = input_flat * target_flat
        intersection = torch.mul(input_flat,target_flat)
 
        num = 2 * intersection.sum() + smooth
        #den = input_flat.sum() + target_flat.sum() + smooth
        den = torch.sum(input_flat.pow(2)) + torch.sum(target_flat.pow(2)) + smooth
        loss = 1 - num / den
        
        #print(num.item(),den.item(),loss.item())
 
        return loss

class Multi_label_DiceLoss(nn.Module):
    def __init__(self):
        super(Multi_label_DiceLoss, self).__init__()
        
        self.dice = DiceLoss()
        self.softmax = nn.Softmax(dim=1)
 
    def	forward(self, inputs, target):
        
#         target = one_hot(target,6)
#         target = target.reshape((target.size()[0],target.size()[4],target.size()[1],target.size()[2],target.size()[3]))
#         target = target.type(torch.FloatTensor)       
#         target = target.cuda()

        target = nn.functional.one_hot(target, 6).permute(0, 4, 1, 2,3).float()       
        logits = self.softmax(inputs)
        
        # print("target:",target.shape)
        # print("logits:",logits.shape)
        total_loss = 0
        C = target.shape[1]
        
        for i in range(C):
            dice_loss = self.dice(logits[:, i], target[:, i])
            total_loss += dice_loss
            # 每个类别的平均 dice_loss
        
        del target,inputs,logits
        
        return total_loss / C
class MultiLabel_Focal_loss(nn.Module):

    def __init__(self,gamma = 0,alpha = 0.8,size_average=True,label_weight = [1,1,1,1,1,1]):
        super(MultiLabel_Focal_loss,self).__init__()
        self.focalloss = Focal_loss_seg(gamma,alpha,size_average)
        self.softmax = nn.Softmax(dim=1)
        self.label_weight = label_weight
    def forward(self, inputs, target):

        target = nn.functional.one_hot(target, 6).permute(0, 4, 1, 2,3).float()       
        logits = self.softmax(inputs)
        total_loss = 0
        C = target.shape[1]
        count = 0
        for i in range(C):
            focal_loss = self.focalloss(logits[:, i], target[:, i])
            labelweight = self.label_weight[i]
            if labelweight!=0:
                total_loss += labelweight * focal_loss
            # 每个类别的平均 dice_loss
                count+=1
        del target,inputs,logits
        if count==0:
            count=1
        return total_loss / count
        
        #fl = -target*self.alpha*((1-inputs)**self.gamma)*torch.log(inputs)-(1-target)*(1-self.alpha)*(inputs**self.gamma)*torch.log(1-inputs)
        


class Multi_label_DiceLoss_pow(nn.Module):
    def __init__(self):
        super(Multi_label_DiceLoss_pow, self).__init__()
        
        self.dice = DiceLoss_pow()
        self.softmax = nn.Softmax(dim=1)
 
    def	forward(self, inputs, target):
        
#         target = one_hot(target,6)
#         target = target.reshape((target.size()[0],target.size()[4],target.size()[1],target.size()[2],target.size()[3]))
#         target = target.type(torch.FloatTensor)       
#         target = target.cuda()

        target = make_one_hot(target,6)        
        logits = self.softmax(inputs)
        
        #print("target:",target.shape)
        #print("logits:",logits.shape)
        total_loss = 0
        C = target.shape[1]
        
        for i in range(C):
            dice_loss = self.dice(logits[:, i], target[:, i])
            total_loss += dice_loss
            # 每个类别的平均 dice_loss
        
        del target,inputs,logits
        
        return total_loss / C

class PDV_LOSS(nn.Module):
    def __init__(self):
        super(PDV_LOSS, self).__init__()
        
        self.dice = DiceLoss()
        
 
    def	forward(self, out1,target):
        
        #target = target.type(torch.LongTensor)
        target = make_one_hot(target,6)   
        
        
        total_loss1 = 0
        
        C = out1.shape[1]
        
        for i in range(C):
            dice_loss1 = self.dice(out1[:, i], target[:, i])
            
            total_loss1 += dice_loss1
            
        
            # 每个类别的平均 dice_loss
        
        
        return total_loss1 / C
    
class PDV_LOSS_pow(nn.Module):
    def __init__(self):
        super(PDV_LOSS_pow, self).__init__()
        
        self.dice = DiceLoss_pow()
        
 
    def	forward(self, out1,target):
        
        #target = target.type(torch.LongTensor)
        target = make_one_hot(target,6)   
        
        
        total_loss1 = 0
        
        C = out1.shape[1]
        
        for i in range(C):
            dice_loss1 = self.dice(out1[:, i], target[:, i])
            
            total_loss1 += dice_loss1
            
        
            # 每个类别的平均 dice_loss
        
        
        return total_loss1 / C
        
class Focal_loss_seg(nn.Module):

    def __init__(self,gamma = 0,alpha = 0.8,size_average=True):
        super(Focal_loss_seg,self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.eps = np.exp(-12)

    def forward(self, inputs, target):


        
        #fl = -target*self.alpha*((1-inputs)**self.gamma)*torch.log(inputs)-(1-target)*(1-self.alpha)*(inputs**self.gamma)*torch.log(1-inputs)
        fl = -target * ((1 - inputs) ** self.gamma) * torch.log(inputs+self.eps) - (1 - target)  * (inputs ** self.gamma) * torch.log(1 - inputs + self.eps)


        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

    


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.cuda()
    result = result.scatter_(1, input, 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target),dim = 1)*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss_new(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]
def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(
        (labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights 
class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0,label_weight = []):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight
        self.label_weight = label_weight
    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_onehot_labels(
            target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        # sigmoid梯度计算
        g = torch.abs(pred.sigmoid().detach() - target)
        # 有效的label的位置
        valid = label_weight > 0
        # 有效的label的数量
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            # 将对应的梯度值划分到对应的bin中， 0-1
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            # 该bin中存在多少个样本
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    # moment计算num bin
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    # 权重等于总数/num bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            # scale系数
            weights = weights / n

        loss = nn.functional.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight
