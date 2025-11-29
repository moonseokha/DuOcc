import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.models.builder import HEADS
from mmdet.models import build_backbone, build_neck
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import build_head, LOSSES


@HEADS.register_module()
class DuOccHead(nn.Module):
    def __init__(
        self,
        streamagg=None,
        queryagg=None,
        mask_decoder_head=None,
        positional_encoding=None,
        det_pred_weight=None,
        group_split=None,
        using_mask="camera",
        occupy_threshold: float = 0.3,
        no_maskhead_infertime: bool = False,
        grid_config=None,
        loss_occ: dict = None,
        train_cfg=None,
        test_cfg=None,
        cls_freq=None,
        occ_pred_weight: float = 1.0,
        use_occ_loss: bool = True,
    ):
        super().__init__()
        
        # Build queryagg and streamagg
        self.queryagg = build_head(queryagg) if isinstance(queryagg, dict) else queryagg
        
        # Build streamagg (config should already include voxel_encoder_backbone, voxel_encoder_neck, img_view_transformer, surround_occ, grid_config)
        self.streamagg = build_from_cfg(streamagg, PLUGIN_LAYERS) if isinstance(streamagg, dict) else streamagg
        
        if group_split is not None:
            self.group_split = torch.tensor(group_split, dtype=torch.uint8)

        if mask_decoder_head is not None:
            mask_decoder_head.update(no_maskhead_infertime=no_maskhead_infertime)
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_test_cfg = test_cfg.pts if test_cfg else None
            mask_decoder_head.update(train_cfg=pts_train_cfg)
            mask_decoder_head.update(test_cfg=pts_test_cfg)
            self.mask_decoder_head = build_head(mask_decoder_head)
        else:
            self.mask_decoder_head = None

        self.loss_occ_mask = nn.BCEWithLogitsLoss()
        self.loss_occ = build_from_cfg(loss_occ, LOSSES) if isinstance(loss_occ, dict) else loss_occ
        if self.loss_occ is not None and cls_freq is not None:
            class_weights = torch.from_numpy(1 / np.log(np.array(cls_freq) + 0.001))
            class_weights[-1] *= 0.5
            class_weights *= 5.0
            self.loss_occ.class_weight = class_weights

        self.positional_encoding = build_positional_encoding(positional_encoding) if isinstance(positional_encoding, dict) else positional_encoding

        self.occ_pred_weight = occ_pred_weight
        self.det_pred_weight = det_pred_weight
        self.using_mask = using_mask
        self.occupy_threshold = occupy_threshold
        self.no_maskhead_infertime = no_maskhead_infertime
        self.grid_config = grid_config  # Keep for build_occ_pos and other uses
        self.use_occ_loss = use_occ_loss


    def forward_train_stage(
        self,
        feature_maps,
        voxel_feature,
        vox_occ_list,
        pred_occ_mask,
        data,
        instance_bank,
        origin_feature_maps,
        occ_pos,
        depths_voxel=None,
    ):
        """
        Query aggregation, detection/mask losses, occupancy losses and bookkeeping.
        Returns:
            output (dict), voxel_feature, up_vox_occ, vox_occ_list
        """
        output = {}
        model_outs, voxel_feature, up_vox_occ = self.queryagg(
            feature_maps=feature_maps, voxel_feature=voxel_feature, metas=data, occ_pos=occ_pos
        )
        instance_bank.cached_vox_feature = voxel_feature.permute(0, 1, 4, 2, 3)
        if "vox_occ" in model_outs:
            vox_occ_list = (vox_occ_list or []) + model_outs["vox_occ"]
        if vox_occ_list is not None:
            output = self.queryagg.loss(model_outs, data)

        if pred_occ_mask is not None and self.loss_occ_mask is not None:
            pred_occ_mask = pred_occ_mask.permute(0, 1, 4, 3, 2).squeeze(1)  # (B, 1, H, W, L) -> (B, L, W, H)
            gt_occ = (data["voxel_semantics"] != 17)  # (B, L, W, H)
            if self.using_mask:
                gt_mask = data["mask_camera"]
            elif self.using_mask == "lidar":
                gt_mask = data["mask_lidar"]
            else:
                gt_mask = torch.ones_like(data["voxel_semantics"])
            weight = torch.where(gt_occ[gt_mask] == 1, 1.0, 0.1)
            output["loss_occ_mask"] = (self.loss_occ_mask(pred_occ_mask[gt_mask], gt_occ[gt_mask].float()) * weight).mean() * 10

        if self.det_pred_weight is not None:
            for key in list(output.keys()):
                output[key] *= self.det_pred_weight

        if self.streamagg is not None and self.streamagg.img_view_transformer is not None and "gt_depth" in data and depths_voxel is not None:
            output["loss_dense_depth"] = self.streamagg.img_view_transformer.get_depth_loss(data["gt_depth"], depths_voxel)

        if self.mask_decoder_head is not None:
            loss_occ, mask_head_occ = self.forward_mask_decoder_train(
                voxel_feature, data, up_vox_occ=up_vox_occ, origin_feature_maps=origin_feature_maps
            )
            if loss_occ is not None:
                output.update(loss_occ)
            if (self.no_maskhead_infertime is not True) and (mask_head_occ is not None):
                if vox_occ_list is None:
                    vox_occ_list = [mask_head_occ.permute(0, 4, 1, 2, 3)]
                else:
                    vox_occ_list.append(mask_head_occ.permute(0, 4, 1, 2, 3))

        if vox_occ_list is not None:
            output_occ = self.loss_occ_pred(
                data["voxel_semantics"], data["mask_camera"], data["mask_lidar"], vox_occ_list, metas=data, using_mask=self.using_mask
            )
            output.update(output_occ)

        instance_bank.cached_vox_feature = instance_bank.cached_vox_feature.detach()
        return output, voxel_feature, up_vox_occ, vox_occ_list

    @force_fp32(apply_to=('voxel_feats'))
    def forward_mask_decoder_train(self,
                          voxel_feats,
                          data,
                          up_vox_occ=None,
                          origin_feature_maps = None,
                          ):
        """
        Forward function for the mask decoder head during training.
        Args:
            voxel_feats (torch.Tensor): [B, C, H, W, Z] 
            data (dict): training data dict
            up_vox_occ: upsampled or processed voxel occupancy from the detection head
            origin_feature_maps: original feature maps for mask decoder
        Returns:
            losses (dict): Mask decoder losses
            outs['occ_outs'] (torch.Tensor): Occupancy predictions
        """
        voxel_semantics = data["voxel_semantics"]
        mask_camera = data["mask_camera"]
        mask_lidar = data["mask_lidar"]

        if self.using_mask is None:
            mask_camera = torch.ones_like(mask_camera)
            mask_lidar = torch.ones_like(mask_lidar)

        gt_classes, sem_mask = self.generate_group(voxel_semantics)

        outs, compact_occ = self.mask_decoder_head(
            voxel_feats, threshold=self.occupy_threshold, full_occ=up_vox_occ, mlvl_feats=origin_feature_maps, **data
        )

        self.queryagg.instance_bank.cached_vox_feature = compact_occ.permute(0, 1, 4, 3, 2)

        loss_inputs = [voxel_semantics, gt_classes, sem_mask,
                       mask_camera, mask_lidar, outs]
        losses = self.mask_decoder_head.loss(*loss_inputs)
        
        return losses,outs['occ_outs']

    def build_occ_pos(self, voxel_size, batch_size, device, dtype):
        """
        Build positional encoding tensor for occupancy grid.
        Returns:
            occ_pos: (B, W*L*H, C) or None if positional_encoding is None
        """
        if self.positional_encoding is None:
            return None

        if voxel_size is None:
            W, L, H = 100, 100, 8
        else:
            W, L, H = voxel_size
        occ_mask = torch.zeros((batch_size, W, L, H), device=device).to(dtype)
        occ_pos = self.positional_encoding(occ_mask, 1).flatten(2).to(dtype).permute(0, 2, 1)
        return occ_pos

    def _voxel_size_from_grid(self):
        if self.grid_config is None:
            return None
        W = int((self.grid_config['x'][1] - self.grid_config['x'][0]) / self.grid_config['x'][2])
        L = int((self.grid_config['y'][1] - self.grid_config['y'][0]) / self.grid_config['y'][2])
        H = int((self.grid_config['z'][1] - self.grid_config['z'][0]) / self.grid_config['z'][2])
        return (W, L, H)

    def forward(
        self,
        feature_maps,
        lift_feature,
        data,
        origin_feature_maps,
        valid_for_detection = False,
    ):
        """
        Unified forward for train/test:
        - voxel encode (view transform + 3D encoder + temporal agg)
        - train: run forward_train_stage and return losses
        - test:  run forward_infer_stage and return outputs
        """
        # Prepare occ_pos for queryagg (needed in forward_train_stage and forward_infer_stage)
        voxel_size = self._voxel_size_from_grid()
        B = lift_feature.shape[0] if hasattr(lift_feature, "shape") else data.get("img").shape[0]
        dtype = lift_feature.dtype if hasattr(lift_feature, "dtype") else torch.float32
        device = lift_feature.device if hasattr(lift_feature, "device") else (data.get("img").device if isinstance(data, dict) and "img" in data else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        occ_pos = self.build_occ_pos(voxel_size, B, device, dtype)

        # Extract instance_bank from queryagg
        instance_bank = self.queryagg.instance_bank if hasattr(self, "queryagg") and self.queryagg is not None else None

        # Voxel feature processing (including streamagg) is now handled by streamagg
        voxel_feature, depths_voxel, vox_occ_list, pred_occ_mask = self.streamagg.voxel_encoder(
            lift_feature, metas=data, instance_bank=instance_bank, training=self.training, data=data, **data
        )
        if self.training:
            output, _, _, _ = self.forward_train_stage(
                feature_maps, voxel_feature, vox_occ_list, pred_occ_mask, data, instance_bank, origin_feature_maps, occ_pos, depths_voxel
            )
            return output
        else:
            output = self.forward_infer_stage(
                feature_maps, voxel_feature, vox_occ_list, data, instance_bank, origin_feature_maps, occ_pos, valid_for_detection=valid_for_detection
            )
            return output

    def forward_infer_stage(
        self,
        feature_maps,
        voxel_feature,
        vox_occ_list,
        data,
        instance_bank,
        origin_feature_maps,
        occ_pos,
        valid_for_detection: bool,
    ):
        """
        Inference stage: query aggregation, optional detection output, occupancy output.
        Returns:
            output (dict)
        """
        output = {}
        occ_results = None
        if vox_occ_list is not None and vox_occ_list != []:
            occ_results = vox_occ_list[-1]

        model_outs, voxel_feature, up_vox_occ = self.queryagg(
            feature_maps=feature_maps, voxel_feature=voxel_feature, metas=data, occ_pos=occ_pos
        )
        instance_bank.cached_vox_feature = voxel_feature.permute(0, 1, 4, 2, 3)

        if "vox_occ" in model_outs:
            occ_results = model_outs["vox_occ"][-1]

        if occ_results is not None:
            occ_score = occ_results.permute(0, 2, 3, 4, 1)
            occ_res = occ_score.argmax(-1)
        else:
            occ_res = None

        if valid_for_detection:
            results = self.queryagg.post_process(model_outs)
            output = {"boxes": results[0]}
        else:
            output = {}

        if self.mask_decoder_head is not None and self.no_maskhead_infertime is not True:
            outs, compact_occ = self.mask_decoder_head(
                voxel_feature, threshold=self.occupy_threshold, full_occ=up_vox_occ, mlvl_feats=origin_feature_maps, **data
            )
            instance_bank.cached_vox_feature = compact_occ.permute(0, 1, 4, 3, 2)
            occ = self.mask_decoder_head.get_occ(outs, occ_res)
            output["occ_results"] = occ["occ"]
        else:
            occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
            output["occ_results"] = occ_res

        return output

    @force_fp32(apply_to=('preds_dicts'))
    def loss_occ_pred(self,
                      gt_occ,  # [B, H, W, L]
                      mask_camera,  # [B, H, W, L]
                      mask_lidar,  # [B, H, W, L]
                      preds_dicts,
                      metas=None,
                      using_mask='camera'
                      ):
        visible_mask = None
        num_total_samples = None
        if using_mask is not None:
            if using_mask == 'camera':
                visible_mask = mask_camera.to(torch.int32)
            elif using_mask == 'lidar':
                visible_mask = mask_lidar.to(torch.int32)
            visible_mask = visible_mask.reshape(-1)
            num_total_samples = visible_mask.sum()

        loss_dict = {}
        voxel_semantics = gt_occ.long()
        for i in range(len(preds_dicts)):
            preds = preds_dicts[i]
            preds = preds.permute(0, 2, 3, 4, 1)
            if self.use_occ_loss and self.loss_occ is not None:
                if using_mask is not None:
                    loss_occ_i = self.loss_occ(preds.reshape(-1, preds.shape[-1]), voxel_semantics.reshape(-1), visible_mask, avg_factor=num_total_samples)
                else:
                    loss_occ_i = self.loss_occ(preds.reshape(-1, preds.shape[-1]), voxel_semantics.reshape(-1))
            else:
                raise NotImplementedError(
                    "Loss functions not configured. Ensure use_occ_loss=True and loss_occ provided."
                )
            loss_occ_i *= self.occ_pred_weight
            if i != (len(preds_dicts) - 1):
                loss_occ_i *= 0.5

            if i == 0:
                loss_name = 'loss_forecast_occ'
            elif i == 1:
                loss_name = 'loss_occ'
            else:
                loss_name = f'loss_pred_occ_{i}'
            loss_dict[loss_name] = loss_occ_i
        return loss_dict

    def generate_mask(self, semantics):
        """Convert semantics to semantic mask for each instance
        Args:
            semantics: [W, H, L]
        Return:
            classes: [N]
                N unique class in semantics
            masks: [N, W, H, L]
                N instance masks
        """
        w, h, z = semantics.shape
        classes = torch.unique(semantics)
        gt_classes = classes.long()
        masks = []
        for class_id in classes:
            masks.append(semantics == class_id)
        if len(masks) == 0:
            masks = torch.zeros(0, w, h, z)
        else:
            masks = torch.stack([x.clone() for x in masks])
        return gt_classes, masks.long()

    def generate_group(self, voxel_semantics):
        """
        Generate grouped classes and masks from voxel semantics.
        This can be used to map fine-grained classes to coarser super-classes.
        Args:
            voxel_semantics: list of [W, H, Z] tensors
        Returns:
            group_classes: List of lists of class indices
            group_masks:   Corresponding binary masks
        """
        group_classes = []
        group_masks = []
        for i in range(len(self.group_split)+1):
            gt_classes = []
            sem_masks = []
            for voxel_semantic in voxel_semantics:
                if not i < 1:
                    w, h, z = voxel_semantic.shape
                    group_split = self.group_split[i-1].to(voxel_semantic)
                    voxel_semantic = group_split[voxel_semantic.flatten().long()].reshape(w, h, z)
                gt_class, sem_mask = self.generate_mask(voxel_semantic)
                gt_classes.append(gt_class.to(voxel_semantic.device))
                sem_masks.append(sem_mask.to(voxel_semantic.device))
            
            group_classes.append(gt_classes)
            group_masks.append(sem_masks)

        return group_classes, group_masks

