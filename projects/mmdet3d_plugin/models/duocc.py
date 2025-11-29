# Copyright (c) 2025, Seokha Moon. All rights reserved.
import torch
import numpy as np
from inspect import signature
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .heads import GridMask

try:
    from ..ops import feature_maps_format
except ImportError:
    feature_maps_format = None

__all__ = ["DuOcc"]


@DETECTORS.register_module()
class DuOcc(BaseDetector):
    def __init__(
        self,
        img_backbone,
        img_neck=None,
        init_cfg=None,
        train_cfg = None,
        test_cfg = None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        num_levels=4,
        valid_for_detection = False,
        multi_neck:dict = None,
        duocc_head: dict = None,
    ):
        """
        Constructor for DuOcc. All major modules and configurations are built here.
        """
        super(DuOcc, self).__init__(init_cfg=init_cfg)

        self.valid_for_detection = valid_for_detection
        self.use_deformable_func = use_deformable_func
        self.num_levels = num_levels

        if multi_neck is not None:
            self.multi_neck = build_neck(multi_neck)
        else:
            self.multi_neck = None

        if pretrained is not None:
            img_backbone.pretrained = pretrained

        self.img_backbone = build_backbone(img_backbone)

        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        else:
            self.img_neck = None

        self.use_grid_mask = use_grid_mask
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
        else:
            self.grid_mask = None

        self.duocc_head = build_head(duocc_head) 


    
    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, metas=None):
        """
        Extract features from the image backbone and neck.
        Args:
            img (torch.Tensor): Input images
            metas (list[dict], optional): Image metadata
        Returns:
            tuple: feature maps from multi-neck or single neck,
                   the lifted feature, and original feature maps for deformable usage
        """
        bs = img.shape[0]
        origin_feature_maps = None

        if img.dim() == 5:
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1

        if self.use_grid_mask:
            img = self.grid_mask(img)

        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)

        if self.img_neck is not None:
            if self.multi_neck is not None:
                lift_feature,inter_feature_maps = self.img_neck(feature_maps[-2:])
                if inter_feature_maps is None:
                    inter_feature_maps =  self.multi_neck(feature_maps)
                else:
                    inter_feature_maps = self.multi_neck(feature_maps[:self.num_levels-2],inter_feature_maps)

                if type(inter_feature_maps)==tuple:
                    inter_feature_maps = [element for element in inter_feature_maps]

                origin_feature_maps = [torch.reshape(feat, (bs, num_cams) + feat.shape[1:]) for feat in inter_feature_maps]
                if self.use_deformable_func and feature_maps_format is not None:
                    inter_feature_maps = feature_maps_format(origin_feature_maps)

            else:
                lift_feature,_ = self.img_neck(feature_maps)
                inter_feature_maps = None
    
        feature_maps = [torch.reshape(lift_feature, (bs, num_cams) + lift_feature.shape[1:])]
        lift_feature = feature_maps[0]

        if self.duocc_head is not None and self.duocc_head.queryagg is not None:
            if self.use_deformable_func and feature_maps_format is not None:
                feature_maps = feature_maps_format(feature_maps)

        if inter_feature_maps is not None:
            return inter_feature_maps, lift_feature, origin_feature_maps

        return feature_maps, lift_feature, origin_feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, lift_feature, origin_feature_maps = self.extract_feat(img, data)
        output = self.duocc_head(
            feature_maps, lift_feature, data, origin_feature_maps,
        )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps, lift_feature, origin_feature_maps = self.extract_feat(img, data)
        output = self.duocc_head(
            feature_maps, lift_feature, data, origin_feature_maps,
            valid_for_detection=self.valid_for_detection,
        )
        return [output]

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)

    def gt_to_voxel(self, gt, vox_shape):
        voxel = np.zeros(vox_shape)
        voxel[gt[:, 0].astype(np.int_), gt[:, 1].astype(np.int_), gt[:, 2].astype(np.int_)] = gt[:, 3]
        return voxel
    def evaluation_semantic(self,pred_occ, gt_occ, class_num):
        results = []

        for i in range(pred_occ.shape[0]):
            gt_i, pred_i = gt_occ[i].cpu().numpy(), pred_occ[i].cpu().numpy()
            gt_i = self.gt_to_voxel(gt_i, pred_occ.shape[1:])
            mask = (gt_i != 255)
            score = np.zeros((class_num, 3))
            for j in range(class_num):
                if j == 0: #class 0 for geometry IoU
                    score[j][0] += ((gt_i[mask] != 0) * (pred_i[mask] != 0)).sum()
                    score[j][1] += (gt_i[mask] != 0).sum()
                    score[j][2] += (pred_i[mask] != 0).sum()
                else:
                    score[j][0] += ((gt_i[mask] == j) * (pred_i[mask] == j)).sum()
                    score[j][1] += (gt_i[mask] == j).sum()
                    score[j][2] += (pred_i[mask] == j).sum()

            results.append(score)
        return np.stack(results, axis=0)