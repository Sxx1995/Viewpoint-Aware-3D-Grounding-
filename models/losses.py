from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    w = torch.clamp(w, min=1e-6)
    h = torch.clamp(h, min=1e-6)
    d = torch.clamp(d, min=1e-6)
    assert (w < 0).sum() == 0
    assert (h < 0).sum() == 0
    assert (d < 0).sum() == 0
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def _volume_par(box):
    return (
        (box[:, 3] - box[:, 0])
        * (box[:, 4] - box[:, 1])
        * (box[:, 5] - box[:, 2])
    )


def _intersect_par(box_a, box_b):
    xA = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
    yA = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
    zA = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
    xB = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
    yB = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
    zB = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
    return (
        torch.clamp(xB - xA, 0)
        * torch.clamp(yB - yA, 0)
        * torch.clamp(zB - zA, 0)
    )


def _iou3d_par(box_a, box_b):
    intersection = _intersect_par(box_a, box_b)
    vol_a = _volume_par(box_a)
    vol_b = _volume_par(box_b)
    union = vol_a[:, None] + vol_b[None, :] - intersection
    return intersection / union, union


def generalized_box_iou3d(boxes1, boxes2):

    assert (boxes1[:, 3:] >= boxes1[:, :3]).all(), boxes1
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = _iou3d_par(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0) 
    volume = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]

    return iou - (volume - union) / volume


class SigmoidFocalClassificationLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input, target):
        loss = (
            torch.clamp(input, min=0) - input * target
            + torch.log1p(torch.exp(-torch.abs(input)))
        )
        return loss

    def forward(self, input, target, weights):
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss
        loss = loss.squeeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


def compute_points_obj_cls_loss_hard_topk(end_points, topk):
    box_label_mask = end_points['box_label_mask']
    seed_inds = end_points['seed_inds'].long() 
    seed_xyz = end_points['seed_xyz']  
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  
    gt_center = end_points['center_label'][:, :, :3]  
    gt_size = end_points['size_gts'][:, :, :3]  
    B = gt_center.shape[0]  
    K = seed_xyz.shape[1]  
    G = gt_center.shape[1]  

    
    point_instance_label = end_points['point_instance_label']  
    obj_assignment = torch.gather(point_instance_label, 1, seed_inds)  
    obj_assignment[obj_assignment < 0] = G - 1  
    obj_assignment_one_hot = torch.zeros((B, K, G)).to(seed_xyz.device)
    obj_assignment_one_hot.scatter_(2, obj_assignment.unsqueeze(-1), 1)

    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1) 
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  
    euclidean_dist1 = (
        euclidean_dist1 * obj_assignment_one_hot
        + 100 * (1 - obj_assignment_one_hot)
    )  
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  

    topk_inds = (
        torch.topk(euclidean_dist1, topk, largest=False)[1]
        * box_label_mask[:, :, None]
        + (box_label_mask[:, :, None] - 1)
    )  
    topk_inds = topk_inds.long()  
    topk_inds = topk_inds.view(B, -1).contiguous()  
    batch_inds = torch.arange(B)[:, None].repeat(1, G*topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([
        batch_inds,
        topk_inds
    ], -1).view(-1, 2).contiguous()

    objectness_label = torch.zeros((B, K + 1)).long().to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)
    objectness_label[objectness_label_mask < 0] = 0

    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(
        seeds_obj_cls_logits.view(B, K, 1),
        objectness_label.unsqueeze(-1),
        weights=cls_weights
    )
    objectness_loss = cls_loss_src.sum() / B

    return objectness_loss


class HungarianMatcher(nn.Module):

    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2,
                 soft_token=False):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        self.soft_token = soft_token

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1) 
        out_bbox = outputs["pred_boxes"].flatten(0, 1) 

        positive_map = torch.cat([t["positive_map"] for t in targets])
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.soft_token:
            if out_prob.shape[-1] != positive_map.shape[-1]:
                positive_map = positive_map[..., :out_prob.shape[-1]]
            cost_class = -torch.matmul(out_prob, positive_map.transpose(0, 1))
        else:
            cost_class = -out_prob[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        cost_giou = -generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(out_bbox),
            box_cxcyczwhd_to_xyzxyz(tgt_bbox)
        )

        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        ).view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64), 
                torch.as_tensor(j, dtype=torch.int64) 
            )
            for i, j in indices
        ]


class SetCriterion(nn.Module):

    def __init__(self, matcher, losses={}, eos_coef=0.1, temperature=0.07):
        super().__init__()
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.temperature = temperature

    def loss_labels_st(self, outputs, targets, indices, num_boxes):
        logits = outputs["pred_logits"].log_softmax(-1)  
        positive_map = torch.cat([t["positive_map"] for t in targets])

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        entropy = torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute bbox losses."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)

        loss_bbox = (
            F.l1_loss(
                src_boxes[..., :3], target_boxes[..., :3],
                reduction='none'
            )
            + 0.2 * F.l1_loss(
                src_boxes[..., 3:], target_boxes[..., 3:],
                reduction='none'
            )
        )
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(src_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['loss_giou_before_sum'] = loss_giou
        losses['giou_belongs'] = idx[0]
        return losses

    def loss_contrastive_align(self, outputs, targets, indices, num_boxes):
        tokenized = outputs["tokenized"]

        norm_text_emb = outputs["proj_tokens"]
        norm_img_emb = outputs["proj_queries"]
        logits = (
            torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
            / self.temperature
        ) 

        positive_map = torch.zeros(logits.shape, device=logits.device)
        inds = tokenized['attention_mask'].sum(1) - 1
        positive_map[torch.arange(len(inds)), :, inds] = 0.5
        positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)
        idx = self._get_src_permutation_idx(indices)
        positive_map[idx] = pmap[..., :logits.shape[-1]]
        positive_map = positive_map > 0

        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        tmask[torch.arange(len(inds)), inds] = 1.0

        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits

        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)
        nb_pos = positive_map.sum(2) + 1e-6
        entropy = -torch.log(nb_pos+1e-6) / nb_pos 
        box_to_token_loss_ = (
            (entropy + pos_term / nb_pos + neg_term)
        ).masked_fill(~boxes_with_pos, 0)
        box_to_token_loss = (box_to_token_loss_ * mask).sum()

        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)
        nb_pos = positive_map.sum(1) + 1e-6
        entropy = -torch.log(nb_pos+1e-6) / nb_pos
        token_to_box_loss = (
            (entropy + pos_term / nb_pos + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        token_to_box_loss = (token_to_box_loss * tmask).sum()

        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        return {"loss_contrastive_align": tot_loss / num_boxes}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def viewpoint_loss(self, outputs, targets, indices, num_boxes, args, **kwargs):
        def loss_calculation(score, gt):
            score = score.transpose(1, 2)
            gt = gt.unsqueeze(0)
            score_ce = (-1 * torch.log(score + 1e-8) * gt) + (-1 * torch.log(1 - score + 1e-8) * (1 - gt))
            best_in_last_layer = score[-1].argmax(-1).unsqueeze(-1)
            acc_count_in_last_layer = torch.gather(gt.squeeze(), index = best_in_last_layer, dim = 1).sum()
            acc_in_last_layer = acc_count_in_last_layer / (gt.squeeze().max(-1)[0].sum() + 1e-8)
            score_ce = (score_ce.mean(0).mean(1) * (gt.squeeze().sum(-1) > 0)).sum() / ((gt.squeeze().sum(-1) > 0).sum() + 1e-8)
            return score_ce, acc_in_last_layer
        loss_vp, vp_last_layer = loss_calculation(outputs["denoise_score"], outputs["sampled_angle"])
        loss_xy, xy_last_layer = loss_calculation(outputs["denoise_xy_score"], outputs["sampled_center"].view(outputs["sampled_center"].shape[0], 100))

        def loss_domain_adaptation(scores, sampled_map):
            loss = - torch.log(scores).squeeze()
            if len(loss.shape) == 3:
               loss = loss.mean(1)
            gt = (sampled_map.squeeze().sum(-1) > 0)
            loss = (loss [:, 0] * gt) + (loss [:, 1] * (1 - gt.float()))
            loss = loss.mean()
            return loss
        loss = loss_vp + loss_xy
        if args.DA_gamma > 0:
            loss_text_DA = loss_domain_adaptation(outputs['text_dis_score'], outputs["sampled_angle"])
            loss_vp_DA = loss_domain_adaptation(outputs['vp_dis_score'], outputs["sampled_angle"])
            loss_vp_xy_DA = loss_domain_adaptation(outputs['vp_xy_dis_score'], outputs["sampled_angle"])
            loss_query_DA = loss_domain_adaptation(outputs['query_dis_score'], outputs["sampled_angle"])
            loss_DA = loss_text_DA + loss_vp_DA + loss_vp_xy_DA + loss_query_DA
            loss = loss + loss_DA
        else:
            loss_DA = torch.zeros([1]).mean()
           
        if args.use_global_obj: 
            tar_G_obj = torch.zeros(outputs['G_obj_dist'].shape[0])
            for idx, i in enumerate(indices):
                if len(i[0]) > 1:
                    tar_G_obj[idx] = -1
                else:
                    tar_G_obj[idx] = i[0][0]
            tar_G_obj = tar_G_obj.unsqueeze(-1).cuda()

            #Dis Loss
            gather_dis = torch.gather(outputs['G_obj_dist'], index = (tar_G_obj * (tar_G_obj >= 0)).long(), dim = 1)
            loss_dis = (gather_dis * (tar_G_obj >= 0)).sum() / ((tar_G_obj >= 0).sum() + 1e-8)
            #Sem Loss
            gather_S_obj_sem_score = torch.gather(outputs['S_obj_sem_score'], index = (tar_G_obj * (tar_G_obj >= 0)).long().unsqueeze(-1).repeat(1, 1, 45), dim = 1).squeeze()
            gather_S_obj_sem_score = torch.gather(gather_S_obj_sem_score, index = ( outputs['obj_name_id'] * ( outputs['obj_name_id'] >= 0)).long().unsqueeze(-1), dim = 1)
            loss_S_sem = (gather_S_obj_sem_score * (tar_G_obj >= 0)).sum() / ((tar_G_obj >= 0).sum() + 1e-8)
            loss_dis = loss_dis + (loss_S_sem) * 0.2
            loss = loss + loss_dis
        else:
            loss_dis = torch.zeros([1]).mean()


        return  {"viewpoint": loss, \
                 "vp_score_prediction": loss_vp.detach(), "angle_prediction_acc": vp_last_layer.detach(), \
                 "xy_score_prediction": loss_xy.detach(), "center_prediction_acc": xy_last_layer.detach(),\
                 "da_prediction": loss_DA.detach(), "global_obj_prediction": loss_dis.detach()}

    def viewpoint_loss_rl(self, outputs, targets, gious, indices, **kwargs):

        def rl_loss_calculation(score, gt, gious, indices):
            giou_mean = 0.5
            gious = gious.detach().data - giou_mean
            gious_r = torch.zeros(score.shape[1]).to(score.device)
            indices = indices.to(score.device)
            for i in range(score.shape[1]):
                gious_r[i] = (gious * (indices == i)).sum() /  (indices == i).sum()
            rl_loss = (gious_r * -1 * torch.log(score[-1].max(-1)[0] + 1e-8)).mean()
            return rl_loss

        rl_loss_vp = rl_loss_calculation(outputs["denoise_score"], \
                                      outputs["sampled_angle"], \
                                      gious, \
                                      indices)
        rl_loss_xy = rl_loss_calculation(outputs["denoise_xy_score"], \
                                      outputs["sampled_center"].view(outputs["sampled_center"].shape[0], 100), \
                                      gious, \
                                      indices)
        rl_loss = rl_loss_vp + rl_loss_xy

        return  {"viewpoint": 0.1 * rl_loss}


    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_st,
            'boxes': self.loss_boxes,
            'contrastive_align': self.loss_contrastive_align,
            'viewpoint': self.viewpoint_loss
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(inds[1]) for inds in indices)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes
            ))

        return losses, indices


def compute_hungarian_loss(end_points, num_decoder_layers, set_criterion, args,
                           query_points_obj_topk=5):
    prefixes = ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    prefixes = ['proposal_'] + prefixes

    gt_center = end_points['center_label'][:, :, 0:3]  
    gt_size = end_points['size_gts']  
    gt_labels = end_points['sem_cls_label']  
    gt_bbox = torch.cat([gt_center, gt_size], dim=-1)  
    positive_map = end_points['positive_map']
    box_label_mask = end_points['box_label_mask']
    target = [
        {
            "labels": gt_labels[b, box_label_mask[b].bool()],
            "boxes": gt_bbox[b, box_label_mask[b].bool()],
            "positive_map": positive_map[b, box_label_mask[b].bool()],
        }
        for b in range(gt_labels.shape[0])
    ]
    loss_ce, loss_bbox, loss_giou, loss_contrastive_align = 0, 0, 0, 0
    for prefix in prefixes:
        output = {}
        if 'proj_tokens' in end_points:
            output['proj_tokens'] = end_points['proj_tokens']
            output['proj_queries'] = end_points[f'{prefix}proj_queries']
            output['tokenized'] = end_points['tokenized']


        pred_center = end_points[f'{prefix}center']  
        pred_size = end_points[f'{prefix}pred_size']  
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        pred_logits = end_points[f'{prefix}sem_cls_scores']  
        output['pred_logits'] = pred_logits
        output["pred_boxes"] = pred_bbox

        output["sampled_angle"] = end_points["sampled_angle"]
        output["sampled_center"] = end_points["sampled_center"]

        output["denoise_score"] = end_points["denoise_score"]
        output["denoise_xy_score"] = end_points["denoise_xy_score"]

        if args.DA_gamma > 0: 
            output["text_dis_score"] = end_points['text_dis_score']
            output["vp_dis_score"] = end_points['vp_dis_score']
            output["vp_xy_dis_score"] = end_points['vp_xy_dis_score']
            output["query_dis_score"] = end_points['query_dis_score']

        if args.use_global_obj:
            output["G_obj_dist"] = end_points['G_obj_dist']
            output["obj_name_id"] = end_points["obj_name_id"]
            output["S_obj_sem_score"] = end_points['S_obj_sem_score']
            output["G_obj_sem_score"] = end_points['G_obj_sem_score']

        losses, indices = set_criterion(output, target)
        for loss_key in losses.keys():
            end_points[f'{prefix}_{loss_key}'] = losses[loss_key]
        loss_ce += losses.get('loss_ce', 0)
        loss_bbox += losses['loss_bbox']
        loss_giou += losses.get('loss_giou', 0)
        if 'proj_tokens' in end_points:
            loss_contrastive_align += losses['loss_contrastive_align']

    view_point_loss = set_criterion.viewpoint_loss(output, target, indices, None, args)
    if args.use_rl:
        rl_view_point_loss = set_criterion.viewpoint_loss_rl(output, target, losses['loss_giou_before_sum'], losses['giou_belongs'])
        view_point_loss['viewpoint'] = view_point_loss['viewpoint'] + rl_view_point_loss['viewpoint']
        losses.pop('loss_giou_before_sum')
        losses.pop('giou_belongs')
    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(
            end_points, query_points_obj_topk
        )
    else:
        query_points_generation_loss = 0.0
    
    # loss
    loss = (
        8 * query_points_generation_loss        
        + args.viewpoint_weight * view_point_loss['viewpoint']
        + 1.0 / (num_decoder_layers + 1) * (    
            loss_ce
            + 7 * loss_bbox                     
            + loss_giou
            + loss_contrastive_align
        )
    )
    end_points['loss_ce'] = loss_ce
    end_points["loss_angle_score_prediction"] = view_point_loss["vp_score_prediction"]
    end_points["loss_center_score_prediction"] = view_point_loss["xy_score_prediction"]
    end_points["loss_da_score_prediction"] = view_point_loss["da_prediction"]
    end_points["loss_global_obj_dis_prediction"] = view_point_loss["global_obj_prediction"]
    end_points["(loss)angle_prediction_acc"] = view_point_loss["angle_prediction_acc"]
    end_points["(loss)center_prediction_acc"] = view_point_loss["center_prediction_acc"]
    end_points['loss_bbox'] = loss_bbox
    end_points['loss_giou'] = loss_giou
    end_points['query_points_generation_loss'] = query_points_generation_loss
    end_points['loss_constrastive_align'] = loss_contrastive_align
    end_points['loss'] = loss
    return loss, end_points
