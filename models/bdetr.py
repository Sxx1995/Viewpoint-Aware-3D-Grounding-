import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast

from .backbone_module import Pointnet2Backbone
from .modules import (
    PointsObjClsModule, GeneralSamplingModule,
    ClsAgnosticPredictHead, PositionEmbeddingLearned
)
from .encoder_decoder_layers import (
    BiEncoder, BiEncoderLayer, BiDecoderLayer
)
from torch.autograd import Function
import copy
import math

class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamda
        return output, None
   
class BeaUTyDETR(nn.Module):

    def __init__(self, num_class=256, num_obj_class=485,
                 input_feature_dim=3,
                 num_queries=256,
                 num_decoder_layers=6, self_position_embedding='loc_learned',
                 contrastive_align_loss=True,
                 d_model=288, butd=True, pointnet_ckpt=None,
                 self_attend=True, gamma = -1, use_global_obj = False):
        """Initialize layers."""
        super().__init__()

        self.use_global_obj = use_global_obj 
        self.gamma = gamma
        self.beta = torch.tensor([0.0001 + (1 - 0.0001) * x / 1000 for x in range(1000)])
        self.alpha = torch.tensor([1 - x for x in self.beta])
        self.acum_alpha = copy.deepcopy(self.alpha)
        for i in range(1, 1000):
            self.acum_alpha[i] = self.acum_alpha[i] * self.acum_alpha[i - 1]
        self.rate_index = np.linspace(0, 550, 6).tolist()[::-1]
        self.beta = self.beta[self.rate_index]
        self.alpha = self.alpha[self.rate_index]
        self.acum_alpha = self.acum_alpha[self.rate_index]
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd

        # Visual encoder
        self.backbone_net = Pointnet2Backbone(
            input_feature_dim=input_feature_dim,
            width=1
        )
        if input_feature_dim == 3 and pointnet_ckpt is not None:
            self.backbone_net.load_state_dict(torch.load(
                pointnet_ckpt
            ), strict=False)

        # Text Encoder
        t_type = "roberta-base"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)
        self.text_encoder = RobertaModel.from_pretrained(t_type)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )

        if self.use_global_obj:
            self.object_list = nn.parameter.Parameter(torch.randn(11000, 288))
            self.disentangle = nn.Linear(288 * 2, 288)
            self.G_classes = nn.Linear(288, 45)

        if self.gamma > 0:
            self.text_selector = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2, eps=1e-12),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 1),
            )

            self.text_discriminator = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2, eps=1e-12),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 2),
                nn.Softmax(dim = -1)
            )
            self.dir_discriminator = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2, eps=1e-12),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 2),
                nn.Softmax(dim = -1)
            )
            self.loc_discriminator = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2, eps=1e-12),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 2),
                nn.Softmax(dim = -1)
            )
            self.bbox_discriminator = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2, eps=1e-12),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 2),
                nn.Softmax(dim = -1)
            )
        # Box encoder
        if self.butd:
            self.butd_class_embeddings = nn.Embedding(num_obj_class, 768)
            saved_embeddings = torch.from_numpy(np.load(
                'data/class_embeddings3d.npy', allow_pickle=True
            ))
            self.butd_class_embeddings.weight.data.copy_(saved_embeddings)
            self.butd_class_embeddings.requires_grad = False
            self.class_embeddings = nn.Linear(768, d_model - 128)
            self.box_embeddings = PositionEmbeddingLearned(24, 128)

        # Cross-encoder
        self.pos_embed = PositionEmbeddingLearned(3, d_model)
        bi_layer = BiEncoderLayer(
            d_model, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=256,
            self_attend_lang=self_attend, self_attend_vis=self_attend,
            use_butd_enc_attn=butd
        )
        self.cross_encoder = BiEncoder(bi_layer, 3)

        # Query initialization
        self.points_obj_cls = PointsObjClsModule(d_model)
        self.gsample_module = GeneralSamplingModule()
        self.decoder_query_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Proposal (layer for size and center)
        self.proposal_head = ClsAgnosticPredictHead(
            num_class, 1, num_queries, d_model,
            objectness=False, heading=False,
            compute_sem_scores=True
        )
        self.directions = nn.Embedding(36, 288)

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.decoder.append(BiDecoderLayer(
                d_model, n_heads=8, dim_feedforward=256,
                dropout=0.1, activation="relu",
                self_position_embedding=self_position_embedding, butd=self.butd
            ))

        # Prediction heads
        self.prediction_heads = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.prediction_heads.append(ClsAgnosticPredictHead(
                num_class, 1, num_queries, d_model,
                objectness=False, heading=False,
                compute_sem_scores=True
            ))

        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )
            self.contrastive_align_projection_text = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )

        # Init
        self.init_bn_momentum()

    def get_box_corner(self, bbox):
        concers = []
        temperate = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]
        for i in range(8):
            for j in range(3):
                concers.append(bbox[:, :, j + temperate[i][j] * 3].unsqueeze(-1))
        return torch.cat(concers, dim = -1)

    def _run_backbones(self, inputs):
        """Run visual and text backbones."""
        # Visual encoder
        end_points = self.backbone_net(inputs['point_clouds'], end_points={})
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = end_points['fp2_xyz']
        end_points['seed_features'] = end_points['fp2_features']
        end_points['encoded_acured_pc_label'] = torch.gather(inputs['acured_pc_label'], dim = -1, index = end_points['seed_inds'].long())
        end_points['seed_xyz'] = end_points['fp2_xyz']
        end_points['seed_features'] = end_points['fp2_features']
         
        # Text encoder
        tokenized = self.tokenizer.batch_encode_plus(
            inputs['text'], padding="longest", return_tensors="pt"
        ).to(inputs['point_clouds'].device)
        encoded_text = self.text_encoder(**tokenized)
        text_feats = self.text_projector(encoded_text.last_hidden_state)
        # Invert attention mask that we get from huggingface
        # because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        end_points['text_feats'] = text_feats
        end_points['text_attention_mask'] = text_attention_mask
        end_points['tokenized'] = tokenized
        return end_points

    def _generate_queries(self, xyz, features, end_points):
        # kps sampling
        points_obj_cls_logits = self.points_obj_cls(features)
        end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
        sample_inds = torch.topk(
            torch.sigmoid(points_obj_cls_logits).squeeze(1),
            self.num_queries
        )[1].int()
        xyz, features, sample_inds = self.gsample_module(
            xyz, features, sample_inds
        )
        end_points['sampled_acured_pc_label'] = torch.gather(end_points['encoded_acured_pc_label'], dim = -1, index = sample_inds.long())
        end_points['query_points_xyz'] = xyz  # (B, V, 3)
        end_points['query_points_feature'] = features  # (B, F, V)
        end_points['query_points_sample_inds'] = sample_inds  # (B, V)
        return end_points
    
    def _generate_location_map(self, xyz, split_num = 10):
        vp_xmax = xyz[:, :, 0].max(1)[0].unsqueeze(1).unsqueeze(1)
        vp_xmin = xyz[:, :, 0].min(1)[0].unsqueeze(1).unsqueeze(1)
        vp_ymax = xyz[:, :, 1].max(1)[0].unsqueeze(1).unsqueeze(1)
        vp_ymin = xyz[:, :, 1].min(1)[0].unsqueeze(1).unsqueeze(1)
        array_x = torch.arange(start = 0, end = split_num).to(xyz.device)
        array_y = torch.arange(start = 0, end = split_num).to(xyz.device)
        array_x = array_x.view(1, split_num, 1)
        array_y = array_y.view(1, 1, split_num)
        array_x = array_x.repeat(xyz.shape[0], 1, 1)
        array_y = array_y.repeat(xyz.shape[0], 1, 1)
        array_x = (vp_xmax - vp_xmin) * array_x / split_num + vp_xmin
        array_y = (vp_ymax - vp_ymin) * array_y / split_num + vp_ymin
        array_x = array_x.repeat(1, 1, split_num)
        array_y = array_y.repeat(1, split_num, 1)
        vp_xy = torch.cat([array_x.unsqueeze(-1), array_y.unsqueeze(-1)], dim = 3)
        return vp_xy.view(xyz.shape[0], -1, 2)

    def get_modified_supervision(self, inputs, angle_index, rotate_matrix):
        tmp = torch.cat([inputs['sampled_angle'], inputs['sampled_angle']], dim = -1)
        new_sampled_angle = torch.zeros(inputs['sampled_angle'].shape).to(inputs['sampled_angle'].device)
        for i in range(inputs['sampled_angle'].shape[0]):
            new_sampled_angle[i] = tmp[i][angle_index[i]:angle_index[i]+inputs['sampled_angle'].shape[1]]
        sampled_center = inputs['sampled_center'].view(inputs['sampled_center'].shape[0], 100)
        return new_sampled_angle.detach(), sampled_center.detach()

    def forward(self, inputs):
        bbox_wrt_concer = self.get_box_corner(inputs['det_boxes'])
        if 'pre_rotate_matrix' in inputs:
            pre_rotate_matrix = inputs['pre_rotate_matrix']
            pre_predict_center = inputs['pre_predict_center']
            pre_rotate_index = inputs['pre_rotate_index']
            vp_xy_ori = inputs['point_clouds'].clone()
            inputs['point_clouds'][:, :, :3] = torch.bmm(inputs['point_clouds'][:, :, :3] - pre_predict_center.unsqueeze(1), pre_rotate_matrix)
            bbox_wrt_concer = torch.bmm(bbox_wrt_concer.view(bbox_wrt_concer.shape[0], bbox_wrt_concer.shape[1] * 8, 3) - pre_predict_center.unsqueeze(1), pre_rotate_matrix)
            bbox_wrt_concer = bbox_wrt_concer.view(bbox_wrt_concer.shape[0], bbox_wrt_concer.shape[1] // 8, 24)
        else:
            pre_rotate_matrix = None 
            pre_predict_center = None

        # Within-modality encoding
        end_points = self._run_backbones(inputs)
        points_xyz = end_points['fp2_xyz']  # (B, points, 3)
        points_features = end_points['fp2_features']  # (B, F, points)
        text_feats = end_points['text_feats']  # (B, L, F)
        text_padding_mask = end_points['text_attention_mask']  # (B, L)

        # Box encoding
        if self.butd:
            # attend on those features
            detected_mask = ~inputs['det_bbox_label_mask']
            detected_feats = torch.cat([
                self.box_embeddings(bbox_wrt_concer),
                self.class_embeddings(self.butd_class_embeddings(
                    inputs['det_class_ids']
                )).transpose(1, 2)  # 92.5, 84.9
            ], 1).transpose(1, 2).contiguous()
        else:
            detected_mask = None
            detected_feats = None

        # Cross-modality encoding

        prob_to_appear = torch.rand(end_points['encoded_acured_pc_label'].size()).cuda()
        prob_to_appear = prob_to_appear < (inputs['epoch'] - 40) / 70
        vp_xy = self._generate_location_map(vp_xy_ori)
        vp_xy = torch.cat([vp_xy, torch.zeros(vp_xy.shape[0], vp_xy.shape[1], 1).to(vp_xy.device)], dim = -1)
        vp_xy = torch.bmm(vp_xy - pre_predict_center.unsqueeze(1), pre_rotate_matrix)
        vp_xy = vp_xy[:, :, 2]
        end_points['encoded_acured_pc_label'] = ((end_points['encoded_acured_pc_label'] + prob_to_appear) > 0).float()
        avoid_empty = (1 - end_points['encoded_acured_pc_label']).sum(-1)
        padding_mask = ((1 - end_points['encoded_acured_pc_label']) * ((avoid_empty).long() < 1024).unsqueeze(-1).float()).bool()
        vp = self.directions.weight.unsqueeze(0).repeat(padding_mask.shape[0], 1, 1)
        denoise_score, denoise_xy_score, vp_feats, vp_xy_feats, points_features, text_feats = self.cross_encoder(
            vp_feats = vp,
            vp_xy = vp_xy,
            vis_feats=points_features.transpose(1, 2).contiguous(),
            pos_feats=self.pos_embed(points_xyz).transpose(1, 2).contiguous(),
            padding_mask = padding_mask,
            text_feats=text_feats,
            text_padding_mask=text_padding_mask,
            end_points=end_points,
            detected_feats=detected_feats,
            detected_mask=detected_mask
        ) 

        end_points['denoise_score'] = denoise_score
        end_points['denoise_xy_score'] = denoise_xy_score
        assert not torch.isnan(points_features.sum()), ((1 - end_points['encoded_acured_pc_label']) * (avoid_empty < 1024).unsqueeze(-1).float()).sum(-1)
        points_features = points_features.transpose(1, 2)
        points_features = points_features.contiguous()  # (B, F, points)
        end_points["text_memory"] = text_feats
        end_points['seed_features'] = points_features
        if self.contrastive_align_loss:
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(text_feats), p=2, dim=-1
            )
            end_points['proj_tokens'] = proj_tokens

        # Query Points Generation
        end_points = self._generate_queries(
            points_xyz, points_features, end_points
        )
        end_points['rotate_index'] = denoise_score[-1].squeeze().argmax(0)
        predicted_angle = denoise_score[-1].squeeze().argmax(0).float() / 36 * 2 * math.pi 
        rotate_matrix = torch.zeros([predicted_angle.shape[0], 3, 3]).to(predicted_angle.device)
        predict_center_idx = denoise_xy_score[-1].squeeze().argmax(0).view(-1, 1, 1).repeat(1, 1, 2)
        predict_center_xy = torch.gather(vp_xy, index = predict_center_idx, dim = 1).squeeze()
        predict_center = torch.zeros([predicted_angle.shape[0], 3]).cuda()
        predict_center[:, :2] = predict_center_xy
 
        rotate_matrix[:, 0, 0] =  torch.cos(predicted_angle)
        rotate_matrix[:, 0, 1] = -torch.sin(predicted_angle)
        rotate_matrix[:, 1, 0] =  torch.sin(predicted_angle)
        rotate_matrix[:, 1, 1] =  torch.cos(predicted_angle)
        rotate_matrix[:, 2, 2] = 1

        end_points['rotate_matrix'] = rotate_matrix
        end_points['predict_center'] = predict_center

        end_points['query_points_xyz'] = torch.bmm(end_points['query_points_xyz'] - predict_center.unsqueeze(1), rotate_matrix)

        if pre_rotate_matrix is not None:
            end_points['sampled_angle'], end_points['sampled_center'] = self.get_modified_supervision(inputs, pre_rotate_index, pre_rotate_matrix)

        cluster_feature = end_points['query_points_feature']  # (B, F, V)
        cluster_xyz = end_points['query_points_xyz']  # (B, V, 3)
        query = self.decoder_query_proj(cluster_feature)
        query = query.transpose(1, 2).contiguous()  # (B, V, F)
        if self.contrastive_align_loss:
            end_points['proposal_proj_queries'] = F.normalize(
                self.contrastive_align_projection_image(query), p=2, dim=-1
            )

        # Proposals (one for each query)
        proposal_center, proposal_size = self.proposal_head(
            cluster_feature,
            base_xyz=cluster_xyz,
            end_points=end_points,
            prefix='proposal_',
            rotate_matrix = rotate_matrix,
            predict_viewpoint_center = predict_center,
            pre_rotate_matrix = pre_rotate_matrix,
            pre_predict_viewpoint_center = pre_predict_center,
        )
        base_xyz = proposal_center.detach().clone()  # (B, V, 3)
        base_size = proposal_size.detach().clone()  # (B, V, 3)
        query_mask = None

        # Box encoding
        if self.butd:
            # attend on those features
            rotated_bbox_wrt_concer = torch.bmm(bbox_wrt_concer.view(bbox_wrt_concer.shape[0], bbox_wrt_concer.shape[1] * 8, 3) - predict_center.unsqueeze(1), rotate_matrix)
            rotated_bbox_wrt_concer = rotated_bbox_wrt_concer.view(rotated_bbox_wrt_concer.shape[0], bbox_wrt_concer.shape[1], 24)
            detected_feats = torch.cat([
                self.box_embeddings(rotated_bbox_wrt_concer),
                self.class_embeddings(self.butd_class_embeddings(
                    inputs['det_class_ids']
                )).transpose(1, 2)  # 92.5, 84.9
            ], 1).transpose(1, 2).contiguous()
        else:
            detected_mask = None
            detected_feats = None
        # Decoder
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if i == self.num_decoder_layers-1 else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError

            # Transformer Decoder Layer
            query = self.decoder[i](
                query, points_features.transpose(1, 2).contiguous(),
                text_feats, query_pos,
                query_mask,
                text_padding_mask,
                detected_feats=(
                    detected_feats if self.butd
                    else None
                ),
                detected_mask=detected_mask if self.butd else None
            )  # (B, V, F)
            if self.contrastive_align_loss:
                end_points[f'{prefix}proj_queries'] = F.normalize(
                    self.contrastive_align_projection_image(query), p=2, dim=-1
                )

            # Prediction
            base_xyz, base_size = self.prediction_heads[i](
                query.transpose(1, 2).contiguous(),  # (B, F, V)
                base_xyz=cluster_xyz,
                end_points=end_points,
                prefix=prefix, 
                rotate_matrix = rotate_matrix,
                predict_viewpoint_center = predict_center,
                pre_rotate_matrix = pre_rotate_matrix,
                pre_predict_viewpoint_center = pre_predict_center,
            )
            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()
        if self.use_global_obj:
            G_obj_gather_id = inputs['G_target_id'] * (inputs['G_target_id'] >= 0)
            select_vp = torch.gather(vp, index = denoise_score[-1].squeeze().argmax(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 288), dim = 1)
            fusioned_feature = torch.cat([select_vp.repeat(1, 256, 1), query], dim = -1)
            fusioned_feature = F.tanh(self.disentangle(fusioned_feature)) 
            G_obj_feat = torch.gather(self.object_list, index = G_obj_gather_id.unsqueeze(-1).repeat(1, 288), dim = 0)
            end_points['G_obj_sem_score'] = F.softmax(self.G_classes(G_obj_feat), dim = -1)
            end_points['S_obj_sem_score'] = F.softmax(self.G_classes(query), dim = -1)
            end_points['G_obj_dist'] = torch.abs(fusioned_feature - G_obj_feat.unsqueeze(1)).mean(-1) - 0.1
            end_points['G_obj_dist'] = end_points['G_obj_dist'] * (end_points['G_obj_dist'] > 0)
        return end_points

    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
