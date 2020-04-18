# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import decode, match, log_sum_exp, point_form


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

##### new loss function definition
    def smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_outside_weights, sigma=1.0 ):
        sigma_2 = sigma ** 2
        outweights = bbox_outside_weights.detach()
        box_diff = bbox_pred - bbox_targets
        in_box_diff = 1.0 * box_diff
        abs_in_box_diff = torch.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
        loss_box = (torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign
                    + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)) * 1.0
        # print(loss_box.size())
        # print(loss_box.shape[0])
        return loss_box.sum() / loss_box.shape[0]

    def KL_loss(self, bbox_pred, bbox_targets, bbox_pred_std, sigma=1.0):
        #KL-loss
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = 1.0 * box_diff #bbox_inw = in_box_diff
        bbox_l1abs = torch.abs(in_box_diff)  #abs_in_box_diff = bbox_l1abs
        # bbox_sq = in_box_diff * in_box_diff
        smoothL1_sign = (bbox_l1abs < 1. / sigma_2).detach().float()  #1 if bbox_l1abs<1 else 0 
        bbox_inws = (torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign
                    + (bbox_l1abs - (0.5 / sigma_2)) * (1. - smoothL1_sign)) 
        bbox_inws = bbox_inws.detach().float()  #?? to be confirmed
        scale = 1
        bbox_pred_std_abs_log = bbox_pred_std*0.5*scale    # 0.5 alpha
        bbox_pred_std_nabs = -1*bbox_pred_std
        bbox_pred_std_nexp = torch.exp(bbox_pred_std_nabs)
        bbox_inws_out = bbox_pred_std_nexp * bbox_inws
        bbox_pred_std_abs_logw = bbox_pred_std_abs_log * 2  # outside weights cancelled
        bbox_pred_std_abs_logwr = torch.mean(bbox_pred_std_abs_logw, dim = 0)
        # print(bbox_pred_std_abs_logw.size())
        # print(bbox_pred_std_abs_logwr.size())
        # print(bbox_pred_std_abs_logwr)

    
        #bbox_pred grad, stop std
        # loss_bbox = self.smooth_l1_loss(bbox_pred, bbox_targets, bbox_pred_std_nexp)
        loss_bbox = F.smooth_l1_loss(bbox_pred, bbox_targets)
        bbox_pred_std_abs_logw_loss = torch.sum(bbox_pred_std_abs_logwr)
        bbox_inws_out = bbox_inws_out * scale
        bbox_inws_outr = torch.mean(bbox_inws_out, dim = 0)
        # print(bbox_inws_out.size())
        # print(bbox_inws_outr.size())
        bbox_pred_std_abs_mulw_loss = torch.sum(bbox_inws_outr)
        return loss_bbox, bbox_pred_std_abs_mulw_loss, bbox_pred_std_abs_logw_loss



    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, std_data, conf_data, priors = predictions
        # print('priors:', priors.size())
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        # print('priors:', priors.size())
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        std_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)   # return ground truth in loc_t
        if self.use_gpu:
            loc_t = loc_t.cuda()   
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)   # loc_t is [ Xmin, Ymin, Xmax, Ymax]
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # print(pos_idx.size())

        loc_p2 = loc_data[pos_idx].view(-1, 4)
        # print(loc_p2.size())
        # print(loc_data[1].size())
        # print(priors.size())
        # print((loc_data[1] + priors).size())
        tensor = torch.ones(())
        coords_box_pred =  tensor.new_empty((loc_data.size()))  #predicted bbox coordinates 
        # print(coords_box_pred.size())
        for i in range(loc_data.size(0)):
            decoded_boxes = decode(loc_data[i], priors, self.variance)  # priors in [x,y,w,h], decoded_boxes in [x,y,w,h]
            coords_box_pred[i] = decoded_boxes

        # print('dd:', coords_box_pred.size())

        # loc_prior = loc_p + priors[pos_idx].view(-1, 4)  # the predicted positions (predicted offsets + priors)
        loc_pre = coords_box_pred[pos_idx].view(-1, 4)           # the predicted positions in [x,y,w,h]
        loc_pre = point_form(loc_pre)             # in [ Xmin, Ymin, Xmax, Ymax]
        loc_t = loc_t[pos_idx].view(-1, 4)        # the ground truth boxes
        std_p = std_data[pos_idx].view(-1, 4)     # the predicted std
        ###################################
        # bbox_in = loc_pre - loc_t
        # bbox_inw = torch.mm(bbox_in * bbox_inside_weights)  # matrix multiplication    #bbox_inside_weights??
        # bbox_l1abs = torch.abs(bbox_inw)
        # bbox_sq = torch.mm(bbox_in * bbox_in)
        # wl1 = torch.ge(bbox_l1abs, 1)
        # wl2 = torch.lt(bbox_l1abs, 1)
        # wl1f = wl1.float()
        # wl2f = wl2.float()
        # bbox_l2_ = torch.mm(bbox_sq, wl2f)
        # bbox_l2 = torch.mul(bbox_l2_, 0.5)

        # bbox_l1abs_ = torch.sub(bbox_l1abs, 0.5)
        # bbox_l1 = torch.mm(bbox_l1abs_, wl1f)
        # bbox_inws = torch.add(bbox_l1, bbox_l2)

        # bbox_pred_std_abs_log = torch.mul(std_p, 0.5)    # 1/2 alpha
        # bbox_pred_std_nabs = torch.mul(std_p, -1)        # -alpha
        # bbox_pred_std_nexp = torch.exp(bbox_pred_std_nabs)      # e^{-alpha}
        # bbox_inws_out = torch.mm(bbox_pred_std_nexp, bbox_inws)
        # #smooth_l1_loss is different in pytorch and in caffe, 
        # loss_bbox = F.smooth_l1_loss(loc_pre, loc_t, size_average=False)
        ###################################
        
        loss_l, kloss1, kloss2 = self.KL_loss(loc_pre, loc_t, std_p)    # return three losses
        print('loss_l:', loss_l)
        print('kloss1:', kloss1)
        print('kloss2:', kloss2)
        ###################################
        # loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1,1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        # loss_l /= N
        # kloss1 /= N
        # kloss2 /= N
        loss_c /= N
        return loss_l, kloss1, kloss2, loss_c

    
