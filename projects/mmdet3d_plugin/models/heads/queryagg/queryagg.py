# Copyright (c) Horizon Robotics. All rights reserved.
# ---------------------------------------------
#  Modified by Seokha Moon
# ---------------------------------------------

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean
from ...modules.matcher import Stage2Assigner
from .blocks import DeformableFeatureAggregation as DFG
__all__ = ["QueryAgg"]

@HEADS.register_module()
class QueryAgg(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        refine_layer: dict,
        ffn: dict = None,
        ffn_vox: dict = None,
        deformable_model: dict = None,
        deformable_model_vox: dict = None,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        dqa_module: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        use_gnn_mask: bool = False,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        allow_low_quality_matches: bool = False,
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: dict = None,
        num_classes: int = 18,
        use_nms_filter: bool = False,
        nms_thr : float = 0.3,
        positional_encoding: dict = None,
        embed_dims: int = 256,
        use_mask_head: bool = False,
        num_img_decoder:int = 5,
        edited_thr: bool = False,
        use_filter: bool = False,
        use_strict_thr: bool = False,
        same_cls_thr: bool = False,
        **kwargs,
    ):
        super(QueryAgg, self).__init__(init_cfg)
        self.use_nms_filter = use_nms_filter
        self.num_img_decoder = num_img_decoder
        self.use_mask_head = use_mask_head
        self.num_decoder = num_decoder
        self.num_classes = num_classes
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.edited_thr = edited_thr
        self.use_strict_thr = use_strict_thr
        self.same_cls_thr = same_cls_thr
        self.use_filter = use_filter
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn
        self.nms_thr = nms_thr
        self.use_gnn_mask = use_gnn_mask
        self.get_cost_mat = Stage2Assigner(allow_low_quality_matches=allow_low_quality_matches,k=1,class_score_thr=0.3)
        self.prediction_length = 0
        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            operation_order = operation_order[3:]
        self.operation_order = operation_order
        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
     
        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.embed_dims = self.instance_bank.embed_dims
        self.num_temp_instances = self.instance_bank.num_temp_instances
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)        
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],

            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        if "ffn" in self.operation_order:
            self.op_config_map.update({
                "ffn": [ffn, FEEDFORWARD_NETWORK],
            })
        if "deformable" in self.operation_order:
            self.op_config_map.update({
                "deformable": [deformable_model, ATTENTION],
            })
        if "deformable_vox" in self.operation_order:
            self.op_config_map.update({
                "deformable_vox": [deformable_model_vox, ATTENTION],
            })
            if "ffn_vox" in self.operation_order:
                self.op_config_map.update({
                    "ffn_vox": [ffn_vox, FEEDFORWARD_NETWORK],
                })
            if "dqa" in self.operation_order:
                self.op_config_map.update({
                    "dqa": [dqa_module, PLUGIN_LAYERS],
                })

        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        voxel_feature: torch.Tensor,
        metas: dict,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        occ_pos: torch.Tensor = None,
        ref_3d: torch.Tensor = None,
    ):
        up_vox_occ = None
        voxel_pos = None
        voxel_feature_occ = None
        if occ_pos is not None:
            voxel_pos = occ_pos

        if spatial_shapes is None:
            if isinstance(feature_maps, torch.Tensor):
                feature_maps = [feature_maps]
            batch_size = feature_maps[0].shape[0]
        else:
            batch_size = feature_maps.shape[2]
        # ========= get instance info ============
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        if self.training:
            (
                instance_feature,
                anchor,
                temp_instance_feature,
                temp_anchor,
                time_interval,
                instance_id,
            ) = self.instance_bank.get(
                batch_size, metas, dn_metas=self.sampler.dn_metas)
        else:
            (
                instance_feature,
                anchor,
                temp_instance_feature,
                temp_anchor,
                time_interval,
                past_id,
            ) = self.instance_bank.get(
                batch_size, metas, dn_metas=self.sampler.dn_metas)
        # ========= prepare for denosing training ============
        # past_id = self.instance_bank.instance_id.clone()
        attn_mask = None
        attn_mask_temp = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            if "instance_id" in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x["instance_id"]).cuda()
                    for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask
            attn_mask_temp = attn_mask.clone()
        else:
            num_instance = instance_feature.shape[1]
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_instance, :num_instance] = False
            attn_mask_temp = attn_mask.clone()
        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # =================== forward the layers ====================
        prediction = []
        classification = []
        quality = []
        anchors = []
        vox_occ_list = [] 
        for i, op in enumerate(self.operation_order):
            if op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask= attn_mask_temp
                    if temp_instance_feature is None
                    else None,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask= attn_mask if self.prediction_length==0 else attn_mask_temp,
                )
            elif op == "norm" or op == "ffn" or op == "ffn_vox":
                instance_feature = self.layers[i](instance_feature)
            elif op == "dqa":
                cls_index = []
                all_filter_flag = [False] * instance_feature.shape[0]
                if self.training:
                    if self.use_nms_filter:
                        reg = prediction[-1].clone().detach()[:,:num_free_instance][..., : len(self.reg_weights)]
                        cls = classification[-1].clone().detach()[:,:num_free_instance]
                        cls_mask = cls.max(-1)[0].sigmoid()>self.nms_thr
                        if self.edited_thr:
                            target_cls,_,_,_,_,_,indices_set= self.sampler.sample(
                                cls.clone(),
                                reg.clone(),
                                metas[self.gt_cls_key],
                                metas[self.gt_reg_key],
                                )
                            if self.use_filter:
                                cls_score = cls.sigmoid()
                                target_cls_score = [cls_score[b][indices_set[b][0]].gather(1,target_cls[b][indices_set[b][0]].unsqueeze(1)).squeeze(1) for b in range(len(target_cls))]
                                for j in range(len(target_cls_score)):
                                    indices_mask = torch.where(target_cls_score[j]>self.nms_thr)[0]
                                    if indices_mask.shape[0] == 0:
                                        all_filter_flag[j] = True
                                        indices = torch.tensor([0])
                                    else:
                                        indices = indices_set[j][0][indices_mask]
                                    cls_index.append(indices)
                            else:
                                cls_max_indices = cls.max(-1)[1]
                                thr_mask = (cls_max_indices == target_cls)
                                for j in range(len(thr_mask)):
                                    indices_mask = torch.where(thr_mask[j][indices_set[j][0]])[0]
                                    if indices_mask.shape[0] == 0:
                                        all_filter_flag[j] = True
                                        indices = torch.tensor([0])
                                    else:
                                        indices = indices_set[j][0][indices_mask]
                                    cls_index.append(indices)
                        else:
                            box_cost= self.sampler.sample(
                                cls.clone(),
                                reg.clone(),
                                metas[self.gt_cls_key],
                                metas[self.gt_reg_key],
                                return_cost=True,
                                )
                            cost_matrix = self.get_cost_mat(cls,reg, metas[self.gt_cls_key],metas[self.gt_reg_key],return_cost_matrix=True) # [pred_ind,gt_ind]
                            for j in range(len(cost_matrix)):
                                if self.use_strict_thr:
                                    indices_mask = torch.where(torch.logical_or(box_cost[j][cls_mask[j]].min(-1)[0] < 0.7,cost_matrix[j][:,cls_mask[j]].max(0)[0]>0.5))[0]
                                elif self.same_cls_thr:
                                    cls_scores = cls.sigmoid()
                                    gt_cls = metas[self.gt_cls_key][j][box_cost[j][cls_mask[j]].min(-1)[1]]
                                    cls_score_mask = cls_scores[j][cls_mask[j]].gather(1,gt_cls.unsqueeze(1)).squeeze() > self.nms_thr
                                    thr_mask = torch.logical_or(box_cost[j][cls_mask[j]].min(-1)[0] < 1.5,cost_matrix[j][:,cls_mask[j]].max(0)[0]>0.3)
                                    indices_mask = torch.where(torch.logical_and( cls_score_mask , thr_mask ))[0]
                                else:
                                    indices_mask = torch.where(torch.logical_or(box_cost[j][cls_mask[j]].min(-1)[0] < 1.5,cost_matrix[j][:,cls_mask[j]].max(0)[0]>0.3))[0]
                                
                                if indices_mask.shape[0] == 0:
                                    all_filter_flag[j] = True
                                    indices = torch.tensor([0],device=voxel_feature.device)
                                else:
                                    indices = torch.where(cls_mask[j])[0][indices_mask]
                                cls_index.append(indices)
                else:
                    num_free_instance = instance_feature.shape[1]
                    if self.use_nms_filter:
                        mask = (classification[-1].sigmoid().max(dim=-1)[0]>self.nms_thr)
                        cls_index = [torch.where(mask)[1]]
                query_feature = instance_feature[:,:num_free_instance].clone()
                query_anchor_embed = anchor_embed[:,:num_free_instance].clone()
                for j,flag_mask in enumerate(all_filter_flag):
                    if flag_mask:
                        query_feature[j] = torch.zeros_like(query_feature[j])
                        query_anchor_embed[j] = torch.zeros_like(query_anchor_embed[j])
                voxel_feature,vox_occ,up_vox_occ= self.layers[i](
                    query_feature,
                    voxel_feature.clone(),
                    anchor[:,:num_free_instance],
                    cls_index,
                    metas= metas,
                    voxel_pos = voxel_pos,
                )

                voxel_feature_occ = voxel_feature.permute(0,4,1,2,3) # [B,H,W,D,C] -> [B,C,H,W,D]
                if vox_occ is not None:
                    vox_occ_list.append(vox_occ)
         
            elif op == "deformable" or op == "deformable_vox":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    voxel_feature,
                    metas,
                    img_feats=feature_maps,
                    spatial_shapes = spatial_shapes,
                    level_start_index = level_start_index,
                )
            elif op == "refine":
                anchors.append(anchor)
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                self.prediction_length += 1
                
                if self.prediction_length == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
         
                if self.operation_order[-1] == "dqa":
                    if i != len(self.operation_order) - 2:
                        anchor_embed = self.anchor_encoder(anchor)
                else:
                    if i != len(self.operation_order) - 1:
                        anchor_embed = self.anchor_encoder(anchor)
                        
                if (self.prediction_length > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}
        output.update({"anchor": anchors})
        if len(vox_occ_list) != 0:
            output.update({"vox_occ": vox_occ_list})

        if dn_metas is not None:
        
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
         
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            # cache dn_metas for temporal denoising
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )    
  


        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )

        self.prediction_length = 0
        if voxel_feature_occ == None:
            voxel_feature_occ = voxel_feature.permute(0,4,1,2,3) # [B,C,H,W,D]
        
        output['instance_feature'] = instance_feature
        return output , voxel_feature_occ, up_vox_occ
    

    
    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}      
        o2o_indices_list = []
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights, _, _ , _, matching_indices= self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            o2o_indices_list.append(matching_indices)
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()
            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(mask, cls.max(dim=-1).values.sigmoid() > threshold)
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]
            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
       
                    
        if "dn_prediction" not in model_outs:
            return output
        

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
        
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )