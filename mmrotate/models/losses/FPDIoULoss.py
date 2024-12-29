import math
import warnings

import cv2
import mmdet.models
import numpy
import numpy as np
import torch
import torch.nn as nn
from mmcv.ops.diff_iou_rotated import box2corners,diff_iou_rotated_2d

from mmdet.structures.bbox import bbox_overlaps

from mmrotate.registry import MODELS
from mmdet.models.losses.utils import weighted_loss



@weighted_loss
def fpdiou_loss(pred, target, linear=False, mode='log', eps=1e-6) -> torch.Tensor:
    """Rotated IoU loss.

    Computing the IoU loss between a set of predicted rbboxes and target
     rbboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn(
            'DeprecationWarning: Setting "linear=True" in '
            'poly_iou_loss is deprecated, please use "mode=`linear`" '
            'instead.')

    if diff_iou_rotated_2d is None:
        raise ImportError('Please install mmcv-full >= 1.5.0.')

    #水平框
    # ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    # res = []
    # for prd in range(len(pred)):
    #     dis=math.pow(pred[prd][0]-target[prd][0],2)+math.pow(pred[prd][1]-target[prd][1],2)+\
    #          math.pow(pred[prd][2] - target[prd][2], 2) + math.pow(pred[prd][3] - target[prd][3], 2)
    #     res.append(dis/(2*1024**2))
    #旋转框
    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    ious = ious.squeeze(0).clamp(min=eps)
    prd_four_points=box2corners(pred.unsqueeze(0))[0].detach().cpu().numpy()
    gt_four_points=box2corners(target.unsqueeze(0))[0].detach().cpu().numpy()
    res=[]
    for prd in range(len(prd_four_points)):
        prd_coords=prd_four_points[prd]
        prd_coords=np.asarray(sorted(prd_coords,key=lambda x:x[0]))
        gt_coords=gt_four_points[prd]
        gt_coords=np.asarray(sorted(gt_coords,key=lambda x:x[0]))

        dis=math.pow(gt_coords[0][0]-prd_coords[0][0],2)+math.pow(prd_coords[0][0]-gt_coords[0][0],2)+\
             math.pow(prd_coords[1][0] - gt_coords[1][0], 2) + math.pow(prd_coords[1][1] - gt_coords[1][1], 2) +\
             math.pow(prd_coords[2][0] - gt_coords[2][0], 2) + math.pow(prd_coords[2][1] - gt_coords[2][1], 2)+\
             math.pow(prd_coords[3][0] - gt_coords[3][0], 2) + math.pow( prd_coords[3][1] - gt_coords[3][1], 2)
        res.append(dis/(4*1024**2))

    res=torch.tensor(res,device=torch.device('cuda:0'))
    if mode == 'linear':
        loss =1-ious+ res
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss



@MODELS.register_module()
class FPDIoULoss(nn.Module):

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='linear'):
        super(MPDIoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 5) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * mpdiou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
