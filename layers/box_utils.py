# -*- coding: utf-8 -*-
import torch
import time


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = matches    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]   # priors [x,y,w,h]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes                                                #[x,y,w,h]?, NO!! actually it is in [x1, y1, x2, y2]!!!


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, std, overlap=0.5, top_k=200, sigma_t=0.02):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    sigma = 0.5
    # simga_2 = tensor.exp(std)
    # print('scores: ', scores.size())
    # print(scores)
    v, idx = scores.sort(descending=True)
    # print('first idx:', idx.size())
    idx = idx[:top_k]
    # print('idx:', idx)
    boxes_new = boxes.new()
    scores_new = scores.new()
    std_new = std.new()
    # print(std.size())
    # print(boxes.size())
    # print(scores.size())
    # print(idx.size())
    torch.index_select(boxes, 0, idx, out=boxes_new)
    torch.index_select(scores, 0, idx, out=scores_new)
    torch.index_select(std, 0, idx, out=std_new)
    # print('std_new:', std_new.size())
    #print(boxes)
    #print('boxes_new before nms:',boxes_new)
    #print('scores_new: before nms',scores_new)
    # print('scores_new:',scores_new.size())
    
    N = boxes_new.size(0) # number of box predictions
    x1 = boxes_new[:, 0]
    y1 = boxes_new[:, 1]
    x2 = boxes_new[:, 2]
    y2 = boxes_new[:, 3]
    areas = torch.mul(x2 - x1, y2 - y1)

    tensor = torch.ones(())
    ious =  torch.zeros((N, N)) 
    kls = torch.zeros((N, N)) 
    # t = time.time()
    for i in range(N):
        xx1 = torch.clamp(x1, min=x1[i])
        yy1 = torch.clamp(y1, min=y1[i])
        xx2 = torch.clamp(x2, max=x2[i])
        yy2 = torch.clamp(y2, max=y2[i])
        w = torch.clamp(xx2 - xx1, min = 0)
        h = torch.clamp(yy2 - yy1, min = 0)
        inter = w*h
        ovr = inter / (areas[i] + areas - inter)
        # print('ovr:',ovr.size())
        # print((ovr.data).cpu().numpy())
        ious[i,:] = ovr
    # print('ious:',ious)
    # print('passed1', time.time()-t)
    # print('N:',N)
    i = 0
    # t = time.time()
    while i < N:
        t3 = time.time()
        maxpos = scores_new[i:N].argmax().data
        maxpos += i
        maxpos = maxpos.item()
        boxes_new[[maxpos,i]] = boxes_new[[i, maxpos]]
        std_new[[maxpos,i]] = std_new[[i,maxpos]]
        scores_new[[maxpos,i]] = scores_new[[i,maxpos]]
        ious[[maxpos,i]] = ious[[i,maxpos]]
        ious[:,[maxpos,i]] = ious[:,[i,maxpos]]
        # print('ious:', ious)

        ovr_bbox = ious[i, i:N].gt(overlap).nonzero()+i
        # ovr_bbox = (ious[i, i:N]*ious[i, i:N].gt(overlap))[0] + i
        # update the boxes predictions
        p = torch.exp(-(1-ious[i, ovr_bbox])**2/sigma_t)
        #print('passed3', time.time()-t3)
        # t4 = time.time()
        BB = boxes_new[ovr_bbox, :]/torch.exp(std_new[ovr_bbox])
        CC = 1./torch.exp(std_new[ovr_bbox])
        # print('BB:', BB)
        # print('CC:', CC)
        # AA = tensor.addcmul(p, BB)
        # print(AA)
        # boxes_new[i,:] = p*(boxes_new[ovr_bbox, :] / torch.exp(std_new[ovr_bbox])) / (p*(1./torch.exp(std_new[ovr_bbox])))
        coordinates1 = tensor.new(len(p),4)
        coordinates2 = tensor.new(len(p),4)
        for ii in range(len(p)):
            # print('pp:',p[ii].item())
            # print(BB[ii])
            # print(CC[ii])
            coordinates1[ii]= p[ii].item()*BB[ii]
            coordinates2[ii]= p[ii].item()*CC[ii]
        # print('coordinates1:', coordinates1.sum(0))
        # print('coordinates2:', coordinates2.sum(0))
        # print('updated coords: ', coordinates1.sum(0)/coordinates2.sum(0))
        boxes_new[i,:] = coordinates1.sum(0)/coordinates2.sum(0)
        # print('boxes_new[i,:]:', boxes_new[i,:])
        #print('passed4', time.time()-t4)

        # t5 = time.time()
        pos = i + 1
        while pos < N:
            if ious[i , pos] > 0:
                ovr =  ious[i , pos]
                scores_new[pos] *= torch.exp(-(ovr * ovr)/sigma)
            if scores_new[pos] < 0.001:
                    boxes_new[[pos, N-1]] = boxes_new[[N-1, pos]]
                    scores_new[[pos, N-1]] = scores_new[[pos, N-1]]
                    std_new[[pos, N-1]] = std_new[[N-1, pos]]
                    ious[[pos, N-1]] = ious[[N-1, pos]]
                    ious[:,[pos, N-1]] = ious[:,[N-1, pos]]
                    N -= 1
                    pos -= 1
            pos += 1
        i += 1
        #print('passed5', time.time()-t5)
    # print('passed2', time.time()-t)
    keep=[i for i in range(N)]
    # print('Last N:', N)
    # print('keep:', keep)
    #print('boxes_new[keep]:', boxes_new[keep])
    #print('scores_new[keep]:', scores_new[keep])
    
    return boxes_new[keep], scores_new[keep], N            # boxes in [keep, 4]         # output after nms, return the selected boxes

    # apply bounding box voting
def box_voting(nms_result, all_result, thresh):  #nms_result in [#, 5], all_result in [#, 5]
    #print('nms_result before voting:', nms_result)
    # print('all_result:', all_result.size())
    # t6 = time.time()
    top_boxes_out = torch.zeros(nms_result.size())
    nms_boxes = nms_result[:, :4]
    all_boxes = all_result[:, :4]
    all_scores = all_result[:, 4]
    #print('all_boxes: ',all_boxes)
    #print('nms_boxes: ',nms_boxes)
    #print('all_scores: ',all_scores)

    nms_to_all_overlaps = bbox_overlaps(nms_boxes, all_boxes)
    #print('nms_to_all_overlaps:', nms_to_all_overlaps)


    for k in range(top_boxes_out.size(0)):
        #print('k:', k)
        inds_to_vote = nms_to_all_overlaps[k].gt(thresh).nonzero()
        # inds_to_vote = np.where(nms_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :].squeeze()
        #print('boxes_to_vote1:', boxes_to_vote)

        ws = all_scores[inds_to_vote].squeeze()
        #print("all_scores",all_scores)
        #print('ws:', ws)
        boxes_to_vote = ws.view(-1,1).mul(boxes_to_vote)
        #print('boxes_to_vote2:', boxes_to_vote)
        #exit()
        try:
            top_boxes_out[k, :4] = torch.mean(boxes_to_vote, dim=0)
        except:
                pass
        top_boxes_out[k, 4] = ws.mean()
        #print('boxes_to_vote3:', top_boxes_out[k, :4])
        #print('boxes_to_vote4:', top_boxes_out[k, 4])
    #print('top_boxes_out after nms: ', top_boxes_out)
    #exit()
    return top_boxes_out


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) 
    query_boxes: (K, 4) 
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.size(0)
    K = query_boxes.size(0)
    overlaps = torch.zeros((N, K))
    print("N,K", N, K)
    for k in range(K):
        # t1 = time.time()
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )
        # print('box_area: ',box_area)
        for n in range(N):
            iw = (
                torch.min(boxes[n, 2], query_boxes[k, 2]) -
                torch.max(boxes[n, 0], query_boxes[k, 0]) 
            )
            if iw > 0:
                ih = (
                    torch.min(boxes[n, 3], query_boxes[k, 3]) -
                    torch.max(boxes[n, 1], query_boxes[k, 1])
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0]) *
                        (boxes[n, 3] - boxes[n, 1]) +
                        box_area - iw * ih
                    )
                    # print('iw: ',iw)
                    # print('ih: ',ih)
                    # print('ua: ',ua)
                    overlaps[n, k] = iw * ih / ua
        # print('each k: ', time.time()-t1)
    return overlaps







def nms2(boxes, scores, std, overlap=0.5, top_k=200, sigma_t=0.02):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
