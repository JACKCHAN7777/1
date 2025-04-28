import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_bbox import dist2bbox, make_anchors


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9, roll_out=False):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 4)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors       = xy_centers.shape[0]
    bs, n_boxes, _  = gt_bboxes.shape
    # Calculate the distance between each real box and the top left and bottom right of each anchors anchor point, and then find the min
    # Make sure that the real frame is near the anchor point, surrounding the anchor point
    if roll_out:
        bbox_deltas = torch.empty((bs, n_boxes, n_anchors), device=gt_bboxes.device)
        for b in range(bs):
            lt, rb          = gt_bboxes[b].view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
            bbox_deltas[b]  = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]),
                                       dim=2).view(n_boxes, n_anchors, -1).amin(2).gt_(eps)
        return bbox_deltas
    else:
        # The real box sits on the bottom right left-top, right-bottom
        lt, rb      = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  
        # The distance between the real box and the top left and bottom right of each anchor point
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # b, n_max_boxes, 8400 -> b, 8400
    fg_mask = mask_pos.sum(-2)
    # If there is an anchor assigned to predict multiple real boxes
    if fg_mask.max() > 1:  
        # b, n_max_boxes, 8400
        mask_multi_gts      = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  
        # If an anchor is assigned to predict multiple real boxes, the real boxes that most coincide with the anchor are first calculated
        # Then make one onehot
        # b, 8400
        max_overlaps_idx    = overlaps.argmax(1)  
        # b, 8400, n_max_boxes
        is_max_overlaps     = F.one_hot(max_overlaps_idx, n_max_boxes)  
        # b, n_max_boxes, 8400
        is_max_overlaps     = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  
        # b, n_max_boxes, 8400
        mask_pos            = torch.where(mask_multi_gts, is_max_overlaps, mask_pos) 
        fg_mask             = mask_pos.sum(-2)
    # Find which gt each anchor matches
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, roll_out_thr=0):
        super().__init__()
        self.topk           = topk
        self.num_classes    = num_classes
        self.bg_idx         = num_classes
        self.alpha          = alpha
        self.beta           = beta
        self.eps            = eps
        # roll_out_thr为64
        self.roll_out_thr   = roll_out_thr

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
            anc_points (Tensor) : shape(num_total_anchors, 2)
            gt_labels (Tensor)  : shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor)  : shape(bs, n_max_boxes, 4)
            mask_gt (Tensor)    : shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor)  : shape(bs, num_total_anchors)
            target_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
            target_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor)        : shape(bs, num_total_anchors)
        """
        # get batch_size
        self.bs             = pd_scores.size(0)
        # Get the maximum number of boxes in the real box
        self.n_max_boxes    = gt_bboxes.size(1)
        # If self.n_max_boxes is greater than self.roll_out_thr then roll_out
        self.roll_out       = self.n_max_boxes > self.roll_out_thr if self.roll_out_thr else False
    
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        # b, max_num_obj, 8400
        # mask_pos      Satisfies the positive sample that is the most coincident between the topk of the real box, and the anchor point that satisfies mask_gt
        # align_metric  The probability that a prior point belongs to a class of a real box multiplied by the degree of overlap between a prior point and the real box
        # overlaps      The degree of overlap between all real boxes and anchor points
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)

        # target_gt_idx     b, 8400     Which gt is suitable for each anchor
        # fg_mask           b, 8400     Does each anchor have a gt that matches it
        # mask_pos          b, max_num_obj, 8400   target_gt_idx after one_hot
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Specify the target to the corresponding anchor point
        # b, 8400
        # b, 8400, 4
        # b, 8400, 80
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Multiply mask_pos and set all anchor points that do not satisfy the real box to 0.
        align_metric        *= mask_pos
        # The maximum score for each real box
        # b, max_num_obj
        pos_align_metrics   = align_metric.amax(axis=-1, keepdim=True) 
        # The maximum overlap for each real box
        # b, max_num_obj
        pos_overlaps        = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        # Multiply the score of each real box and a prior point by the maximum overlap, and divide the maximum score
        norm_align_metric   = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # target_scores as regular tag
        target_scores       = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # pd_scores bs, num_total_anchors, num_classes
        # pd_bboxes bs, num_total_anchors, 4
        # gt_labels bs, n_max_boxes, 1
        # gt_bboxes bs, n_max_boxes, 4

        # align_metric is a calculated generation value. The probability of a class whose prior point belongs to a real box
        # is multiplied by the degree of overlap between a prior point and the real box.
        # overlaps is the degree to which a priori point coincides with the real box
        # align_metric, overlaps    bs, max_num_obj, 8400
        align_metric, overlaps  = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        
       # Positive sample anchors need to meet at the same time:
        # 1. In the real box
        # 2. It is the most overlapping positive sample of the real box topk
        # 3. Satisfies mask_gt
        
        # get in_gts mask           b, max_num_obj, 8400
        # Determine whether the prior point is in the real box
        mask_in_gts             = select_candidates_in_gts(anc_points, gt_bboxes, roll_out=self.roll_out)
        # get topk_metric mask      b, max_num_obj, 8400
        # Determine whether the anchor is in the topk of the real box
        mask_topk               = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask, b, max_num_obj, h*w
        # Real box exists, not padding
        mask_pos                = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        if self.roll_out:
            align_metric    = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            overlaps        = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            ind_0           = torch.empty(self.n_max_boxes, dtype=torch.long)
            for b in range(self.bs):
                ind_0[:], ind_2 = b, gt_labels[b].squeeze(-1).long()
                # Obtain scores in this category
                # bs, max_num_obj, 8400
                bbox_scores     = pd_scores[ind_0, :, ind_2]  
                # Calculate real and predicted boxes ciou
                # bs, max_num_obj, 8400
                overlaps[b]     = bbox_iou(gt_bboxes[b].unsqueeze(1), pd_bboxes[b].unsqueeze(0), xywh=False, CIoU=True).squeeze(2).clamp(0)
                align_metric[b] = bbox_scores.pow(self.alpha) * overlaps[b].pow(self.beta)
        else:
            # 2, b, max_num_obj
            ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)       
            # b, max_num_obj  
            # [0]Represents the picture
            ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  
            # [1]What is the label
            ind[1] = gt_labels.long().squeeze(-1) 
            # Obtain scores in this category
            # Take out the probability that a priori point belongs to a certain class
            # b, max_num_obj, 8400
            bbox_scores = pd_scores[ind[0], :, ind[1]]  

            # Calculate real and predicted boxes ciou
            # bs, max_num_obj, 8400
            overlaps        = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
            align_metric    = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Args:
            metrics     : (b, max_num_obj, h*w).
            topk_mask   : (b, max_num_obj, topk) or None
        """
        # 8400
        num_anchors             = metrics.shape[-1] 
        # b, max_num_obj, topk
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # b, max_num_obj, topk
        topk_idxs[~topk_mask] = 0
        # b, max_num_obj, topk, 8400 -> b, max_num_obj, 8400
        # This step results in is_in_topk is b, max_num_obj, 8400
        # represents the top k prior points corresponding to each real box
        if self.roll_out:
            is_in_topk = torch.empty(metrics.shape, dtype=torch.long, device=metrics.device)
            for b in range(len(topk_idxs)):
                is_in_topk[b] = F.one_hot(topk_idxs[b], num_anchors).sum(-2)
        else:
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # Determine whether the anchor is in the topk of the real box
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels       : (b, max_num_obj, 1)
            gt_bboxes       : (b, max_num_obj, 4)
            target_gt_idx   : (b, h*w)
            fg_mask         : (b, h*w)
        """

        # Used to read real box labels, (b, 1)
        batch_ind       = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        # b, h*w    Get gt_labels, gt_bboxes' sequence number after flatten
        target_gt_idx   = target_gt_idx + batch_ind * self.n_max_boxes
        # b, h*w    Used to read tags after flatten
        target_labels   = gt_labels.long().flatten()[target_gt_idx]
        # b, h*w, 4 Used to read box after flatten
        target_bboxes   = gt_bboxes.view(-1, 4)[target_gt_idx]
        
        # assigned target scores
        target_labels.clamp(0)
        # Perform one_hot mapping to the form required for training.
        target_scores   = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask  = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores   = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)

class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # Calculate IOU Loss
        # weight represents the confidence that the tag should have in the loss, 0 is the minimum and 1 is the maximum
        weight      = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        #Calculate the degree of overlap between the prediction box and the real box
        iou         = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # Then 1-coin degree, multiply the confidence you should have, and then sum to find the average.
        loss_iou    = ((1.0 - iou) * weight).sum() / target_scores_sum

        # Calculate DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        # A point generally does not lie at an anchor point, usually xx.xx. If you want to use DFL, it is impossible to fit it directly with a cross_entropy
        # So consider it as the distance between the upper left anchor point of xx.xx and the lower right anchor point. If the distance from the lower right anchor point is small, wl will be small, and the loss in the upper left corner will be small.
        # If the distance from the upper left corner is small, wr will be small, and the loss in the lower right corner will be small.
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

# Criterion class for computing training losses
class Loss:
    def __init__(self, model): 
        self.bce    = nn.BCEWithLogitsLoss(reduction='none')
        self.stride = model.stride  # model strides
        self.nc     = model.num_classes  # number of classes
        self.no     = model.no
        self.reg_max = model.reg_max
        
        self.use_dfl = model.reg_max > 1
        roll_out_thr = 64

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0,
                                            roll_out_thr=roll_out_thr)
        self.bbox_loss  = BboxLoss(model.reg_max - 1, use_dfl=self.use_dfl)
        self.proj       = torch.arange(model.reg_max, dtype=torch.float)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=targets.device)
        else:
            # Obtain image index
            i           = targets[:, 0]  
            _, counts   = i.unique(return_counts=True)
            out         = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
            # Loop the batch and assign the value
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            # Zoom to original image size.
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            # batch, anchors, channels
            b, a, c     = pred_dist.shape  
            # Decoding of DFL
            pred_dist   = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.to(pred_dist.device).type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        # Then decode to obtain the prediction box
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        # Obtain the device used
        device  = preds[1].device
        # box, cls, dfl Three part loss
        loss    = torch.zeros(3, device=device)  
        # Obtain characteristics and divide them
        feats   = preds[2] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)

        # bs, num_classes + self.reg_max * 4 , 8400 =>  cls bs, num_classes, 8400; 
        #                                               box bs, self.reg_max * 4, 8400
        pred_scores = pred_scores.permute(0, 2, 1).contipansus()
        pred_distri = pred_distri.permute(0, 2, 1).contipansus()

        # Get batch size and dtype
        dtype       = pred_scores.dtype
        batch_size  = pred_scores.shape[0]
        # Get input image size
        imgsz       = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]  
        # Obtain the tensor corresponding to anchors points and step size
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Make a matrix of things in a batch
        # 0 is the picture of the first one
        #1 is the category
        #2: The coordinates of the box
        targets                 = torch.cat((batch[:, 0].view(-1, 1), batch[:, 1].view(-1, 1), batch[:, 2:]), 1)
        # First perform preliminary processing, padding the input gt to the maximum number, and scale the coordinates of the box
        # bs, max_boxes_num, 5
        targets                 = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # bs, max_boxes_num, 5 => bs, max_boxes_num, 1 ; bs, max_boxes_num, 4
        gt_labels, gt_bboxes    = targets.split((1, 4), 2)  # cls, xyxy
        # Find which boxes have goals and which ones are filled
        # bs, max_boxes_num
        mask_gt                 = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        # Decode the prediction results and obtain the prediction box
        # bs, 8400, 4
        pred_bboxes             = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # Assign prediction and real boxes
        # target_bboxes     bs, 8400, 4
        # target_scores     bs, 8400, 80
        # fg_mask           bs, 8400
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )

        target_bboxes       /= stride_tensor
        target_scores_sum   = max(target_scores.sum(), 1)

        # Calculate the loss of the classification
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Calculate the loss of the bbox
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        return loss.sum() # loss(box, cls, dfl) # * batch_size

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
