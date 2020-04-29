import torch
from torch.autograd import Function
from ..box_utils import decode, nms, box_voting, nms2, nms_no_voting
from data import voc as cfg


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, std_data, conf_data, prior_data):

        # print('kkk')
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)                          # batch size
        
        num_priors = prior_data.size(0)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        final_output = torch.zeros(num, self.num_classes, self.top_k, 5)

        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)
        # print(loc_data.size())
        # print(std_data.size())
        # print(conf_data.size())
        # print(prior_data.size())

        # print('num:', num)
        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)   # in [x1, y1, x2, y2]
            # print('decoded_boxes: ',decoded_boxes.size())
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            # print('conf_scores: ',conf_scores.size())
            # print('conf_scores: ',conf_scores)
            # for cl in range(1, self.num_classes):
            for cl in range(1, conf_preds.size(1)):
            #for cl in range(1, 3):
                # print('class num: ', cl)
                # print()
                # print(torch.mean(conf_scores))
                # print(torch.std(conf_scores))
                # exit()
                c_mask = conf_scores[cl].gt(torch.mean(conf_scores).item()*16)
                # print('conf_scores: ',conf_scores[cl])
                # c_mask = conf_scores[cl].gt(0.02)                 ####### self.conf_thresh changed
                # print('c_mask: ',c_mask.size())
                scores = conf_scores[cl][c_mask]
                # print('scores: ',scores)
                # print('scores: ',c_mask.nonzero())
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                std = std_data[i][l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                # print('before nms: ',boxes.size())
                # print(scores.size())
                # print(std.size())

                # nms_boxes, nms_scores, count = nms(boxes, scores, std, self.nms_thresh, self.top_k)

                nms_boxes, nms_scores, count = nms_no_voting(boxes, scores, std, self.nms_thresh, self.top_k)

                nms_boxes_scores = torch.cat((nms_boxes, nms_scores.view(-1,1)), 1)
                all_boxes_scores = torch.cat((boxes, scores.view(-1,1)), 1)
                # print('before voting: ',all_boxes_scores.size())

                #final_boxes = box_voting(nms_boxes_scores, all_boxes_scores, 0.45)         ######## voting thresh?
                
                #print('final_boxes: ',final_boxes.size())

                #final_boxes_boxes = final_boxes[:,:4]
                #final_boxes_scores = final_boxes[:,4]
                # print('final_boxes_scores: ',final_boxes_scores.size())
                # print('final_boxes_boxes: ',final_boxes_boxes.size())

                #output[i, cl, :count] = torch.cat((final_boxes_scores.view(-1,1),final_boxes_boxes),1)    # boxes after voting
                
                # print('final_boxes: ',final_boxes.size())

                # output[i, cl, :count] = torch.cat((nms_scores.view(-1,1),nms_boxes), 1)    # boxes after voting

                # output[i, cl, :count] = final_boxes    # boxes after voting
                
                # print(output.size())
                # print(torch.cat((scores.view(-1,1), boxes),1).size())
                output[i, cl, :count] = torch.cat((nms_scores.view(-1,1), nms_boxes),1)
            # print('decoded: ', decoded_boxes)
            # print('scores: ', scores)
                # ids, count = nms2(boxes, scores, self.nms_thresh, self.top_k)
                # output[i, cl, :count] = \
                #     torch.cat((scores[ids[:count]].unsqueeze(1),
                #                boxes[ids[:count]]), 1)

                
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
