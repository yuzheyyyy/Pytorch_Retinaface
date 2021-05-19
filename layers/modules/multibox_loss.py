

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg_mnet
GPU = cfg_mnet['gpu_train']

class Wing(nn.Module):
    def __init__ (self, omega=1, epsilon=1.54):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, inputs, target):
        lossMat = torch.zeros_like(inputs)
        case1_ind = torch.abs(inputs-target)<self.omega
        case2_ind = torch.abs(inputs-target)>=self.omega
        lossMat[case1_ind]=torch.log(1+torch.abs(inputs[case1_ind]-target[case1_ind])/self.epsilon)
        lossMat[case2_ind]=torch.abs(inputs[case2_ind]-target[case2_ind])-0.5
        return lossMat

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

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.wing = Wing()

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data, visible_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)

        angle_t = torch.LongTensor(num, num_priors)
        visible_t = torch.Tensor(num, num_priors, 5)
        # euler_t = torch.Tensor(num, num_priors, 3)

        for idx in range(num):

            '''
            for label with angle
            '''
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -7].data
            landms = targets[idx][:, 4:14].data
            angles = targets[idx][:, -6].data
            visible = targets[idx][:, -5:].data


            defaults = priors.data

            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx, angles, angle_t, visible, visible_t)


        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

            angle_t = angle_t.cuda()
            visible_t = visible_t.cuda()

        zeros = torch.tensor(0).cuda()
        ang_thr = torch.tensor(60).cuda()

        pos1 = conf_t > zeros

        num_pos_landm = pos1.long().sum(1, keepdim=True)

        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)

    
        mask_angle = angle_t > ang_thr
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)

         

        # # baseline
        landm_p = landm_data[pos_idx1].view(-1, 10)  
        landm_t = landm_t[pos_idx1].view(-1, 10)

        # HEM
        mask = (landm_t == -1) # we only calculate the loss of visible landmarks
        mask = torch.logical_not(mask)
        # loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='none')

        # wing loss
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='none')
        loss_landm_mask = mask * loss_landm
        loss_landm_mask_mean = torch.mean(loss_landm_mask, -1)
        loss_landm_mask_sum = torch.sum(loss_landm_mask, -1)
        # print('size {}'.format(loss_landm_mask.shape))
        size = int(0.5 * loss_landm_mask.shape[0])
        _, topk_idx = torch.topk(loss_landm_mask_mean, k=size)
        loss_landm = torch.sum(loss_landm_mask_sum[topk_idx])
        N2 = size

     

        vis_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(visible_data)
        vis_p = visible_data[vis_idx1].view(-1, 5)
        vis_t = visible_t[vis_idx1].view(-1, 5)
        vis_p = torch.sigmoid(vis_p)
  
        criterions = nn.BCELoss(reduction='sum')
        loss_vis = criterions(vis_p, vis_t)




        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)


        ang_pos = mask_angle.unsqueeze(mask_angle.dim()).expand_as(loc_data)
        ang_not_pos = torch.logical_not(ang_pos)


        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        loss_l = torch.sum(loss_l)


        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)


        ang_conf_idx = mask_angle.unsqueeze(mask_angle.dim()).expand_as(conf_data)
        ang_not_conf_idx = torch.logical_not(ang_conf_idx)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)

        loss_l /= N
        loss_c /= N
        loss_landm /= N2
        loss_vis /= N1


        return loss_l, loss_c, loss_landm, loss_vis