# Copyright Seokha Moon. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.cnn import build_conv_layer, build_upsample_layer, build_norm_layer
from mmcv.runner import force_fp32
from mmcv.utils import build_from_cfg

from mmdet.models import build_backbone, build_neck
from mmcv.cnn import Linear
__all__ = ["StreamAgg"]

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

@PLUGIN_LAYERS.register_module()
class StreamAgg(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 refine_net_cfg: dict,
                 use_forecast_head: bool = False,
                 grid_config = None,
                 temp_cat_method: str = 'add',
                 num_classes = 17,
                 voxel_encoder_backbone=None,
                 voxel_encoder_neck=None,
                 img_view_transformer=None,
                 surround_occ: bool = False,
                 ):
        super(StreamAgg, self).__init__()
        self.embed_dims = embed_dims
        self.use_forecast_head = use_forecast_head
        self.temp_cat_method = temp_cat_method
        self.surround_occ = surround_occ
        self.grid_config = grid_config
        
        # Build nested modules from configs if dicts are provided
        self.voxel_encoder_backbone = build_backbone(voxel_encoder_backbone) if isinstance(voxel_encoder_backbone, dict) else voxel_encoder_backbone
        self.voxel_encoder_neck = build_neck(voxel_encoder_neck) if isinstance(voxel_encoder_neck, dict) else voxel_encoder_neck
        self.img_view_transformer = build_from_cfg(img_view_transformer, PLUGIN_LAYERS) if isinstance(img_view_transformer, dict) else img_view_transformer
        
        conv3d_cfg=dict(type='Conv3d', bias=False)
        gn_norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)

        # if use_temporal:
        self.temporal_conv_net = build_backbone(refine_net_cfg)
        if self.temp_cat_method == 'cat':
            conv = build_conv_layer(conv3d_cfg, embed_dims*2, embed_dims, kernel_size=1, stride=1)
            self.cat_block = nn.Sequential(conv,
                            build_norm_layer(gn_norm_cfg, embed_dims)[1],
                            nn.ReLU(inplace=True))      

        if use_forecast_head:
            deconv_cfg = dict(type='deconv3d', bias=False)
            out_dims = embed_dims

            pred_upsample = build_upsample_layer(deconv_cfg, embed_dims, out_dims, kernel_size=2, stride=2)
            self.pred_up_block = nn.Sequential(pred_upsample,
                            nn.BatchNorm3d(out_dims),
                            nn.ReLU(inplace=True))
            self.forecast_occ_net = build_conv_layer(
                conv3d_cfg,
                in_channels=out_dims,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0)
           
        if grid_config is not None:
            self.bev_origin = [grid_config['x'][0], grid_config['y'][0], grid_config['z'][0]]
            self.grid_size = [int((grid_config['x'][1]-grid_config['x'][0])/grid_config['x'][2]), int((grid_config['y'][1]-grid_config['y'][0])/grid_config['y'][2]), int((grid_config['z'][1]-grid_config['z'][0])/grid_config['z'][2])]
            self.bev_resolution = [(grid_config['x'][1]-grid_config['x'][0])/self.grid_size[0], (grid_config['y'][1]-grid_config['y'][0])/self.grid_size[1], (grid_config['z'][1]-grid_config['z'][0])/self.grid_size[2]]  
            self.grid_range = [grid_config['x'][1]-grid_config['x'][0], grid_config['y'][1]-grid_config['y'][0], grid_config['z'][1]-grid_config['z'][0]]

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_voxel_backbone(self, voxel_feat):
        if self.voxel_encoder_backbone is not None:
            voxel_feat = self.voxel_encoder_backbone(voxel_feat)
        if self.voxel_encoder_neck is not None:
            voxel_feat, _ = self.voxel_encoder_neck(voxel_feat)
        if isinstance(voxel_feat, (tuple, list)):
            voxel_feat = voxel_feat[0]
        return voxel_feat

    def lightweight_fpn(self, x, metas):
        """
        Lightweight FPN: view transform + voxel backbone/neck encoding
        Args:
            x: Input features
            metas: Metadata dictionary
        Returns:
            voxel_feat: Encoded voxel features
            lss_depth: Depth features from LSS view transformer
        """
        mlp_input = self.img_view_transformer.get_mlp_input(metas["view_tran_comp"])
        voxel_feat, lss_depth = self.img_view_transformer([x] + metas["view_tran_comp"], metas["projection_mat"], mlp_input)
        voxel_feat = self.encode_voxel_backbone(voxel_feat)
        return voxel_feat, lss_depth

    @torch.no_grad()
    def _build_flow_grid(self, grid_size, device, dtype):
        if self.grid_config is not None:
            xs = torch.linspace(self.grid_config["x"][0], self.grid_config["x"][1], grid_size[1], device=device)
            ys = torch.linspace(self.grid_config["y"][0], self.grid_config["y"][1], grid_size[2], device=device)
            zs = torch.linspace(self.grid_config["z"][0], self.grid_config["z"][1], grid_size[0], device=device)
        else:
            xs = torch.linspace(-1, 1, grid_size[1], device=device)
            ys = torch.linspace(-1, 1, grid_size[2], device=device)
            zs = torch.linspace(-1, 1, grid_size[0], device=device)

        Z, Y, X = torch.meshgrid(zs, ys, xs)
        grid = torch.stack([X, Y, Z], dim=-1).unsqueeze(0).to(dtype=dtype)
        return grid

    def forward_voxel_stage(self, voxel_feat, metas, instance_bank, training: bool, **kwargs):
        """
        Stream aggregation, temporal warping, caching, and output permutation for voxel features.
        Returns:
            voxel_feat_out: (B, W, L, H, C)
            vox_occ_list: list or None
            pred_occ_mask: tensor or None
        """
        if instance_bank is not None:
            if (
                instance_bank.cached_anchor is not None
                and voxel_feat.shape[0] != instance_bank.cached_anchor.shape[0]
            ):
                instance_bank.reset_vox_feature()
                instance_bank.metas = None
            prev_vox_feat = instance_bank.cached_vox_feature
            prev_metas = instance_bank.metas
        else:
            prev_vox_feat = None
            prev_metas = None

        vox_occ_list = None
        pred_occ_mask = None

        if prev_vox_feat is None:
            prev_vox_feat = voxel_feat.clone()
            prev_metas = None
        
        grid_size = prev_vox_feat.shape[2:]
        grid = self._build_flow_grid(grid_size, device=voxel_feat.device, dtype=voxel_feat.dtype)
        _, d, h, w, c = grid.shape
        grid = grid.view(1, d, h, w, c).expand(voxel_feat.shape[0], d, h, w, c)  # (B, H, W, L, 3)

        if prev_metas is not None:
            prev_times = prev_metas["timestamp"]
            prev_metas_img = prev_metas["img_metas"]
        else:
            prev_times = None
            prev_metas_img = None

        if prev_metas_img is not None and metas is not None and "img_metas" in metas and metas["img_metas"] is not None:
            if self.surround_occ:
                T_temp2cur = torch.tensor(
                    np.stack(
                        [
                            prev_metas_img[i]["lidar2ego_inv"]
                            @ prev_metas_img[i]["T_global_inv"]
                            @ m["T_global"]
                            @ m["lidar2ego"]
                            for i, m in enumerate(metas["img_metas"])
                        ]
                    ),
                    device=voxel_feat.device,
                    dtype=voxel_feat.dtype,
                )  # current to previous [B,4,4]
            else:
                T_temp2cur = torch.tensor(
                    np.stack(
                        [
                            prev_metas_img[i]["T_global_inv"]  # global to ego
                            @ m["T_global"]  # ego to global
                            for i, m in enumerate(metas["img_metas"])
                        ]
                    ),
                    device=voxel_feat.device,
                    dtype=voxel_feat.dtype,
                )
        else:
            T_temp2cur = torch.eye(4, device=voxel_feat.device, dtype=voxel_feat.dtype).unsqueeze(0).repeat(voxel_feat.shape[0], 1, 1)

        grid = torch.matmul(T_temp2cur[:, None, None, None, :3, :3], grid[..., None]).squeeze(-1) + T_temp2cur[
            :, None, None, None, :3, 3
        ]

        if self.grid_config is not None:
            grid[..., 0] -= (self.grid_config["x"][0] + self.grid_config["x"][1]) / 2
            grid[..., 1] -= (self.grid_config["y"][0] + self.grid_config["y"][1]) / 2
            grid[..., 2] -= (self.grid_config["z"][0] + self.grid_config["z"][1]) / 2
            grid[..., 0] /= (self.grid_config["x"][1] - self.grid_config["x"][0]) / 2
            grid[..., 1] /= (self.grid_config["y"][1] - self.grid_config["y"][0]) / 2
            grid[..., 2] /= (self.grid_config["z"][1] - self.grid_config["z"][0]) / 2

        prev_vox_feat = F.grid_sample(prev_vox_feat, grid, align_corners=True)

        if prev_times is not None and metas is not None and "timestamp" in metas:
            time_interval = metas["timestamp"] - prev_times
            time_interval = time_interval.to(dtype=voxel_feat.dtype)
            mask = torch.logical_and(torch.abs(time_interval) <= 2.0, time_interval != 0)
        else:
            mask = [True for _ in range(voxel_feat.shape[0])]

        for i, m in enumerate(mask):
            if not m:
                if prev_vox_feat.shape[1] != voxel_feat.shape[1]:
                    if prev_vox_feat is not None:
                        prev_vox_feat[i] = prev_vox_feat[i].new_zeros(prev_vox_feat[i].shape)
                else:
                    if training:
                        prev_vox_feat[i] = voxel_feat[i].clone().detach()
                    else:
                        prev_vox_feat[i] = voxel_feat[i]

        voxel_feat, vox_occ_list, pred_occ_mask = self.forward(voxel_feat, prev_vox_feat, **kwargs)

        if instance_bank is not None:
            instance_bank.cached_vox_feature = voxel_feat.clone()
        voxel_feat = voxel_feat.permute(0, 3, 4, 2, 1)
        return voxel_feat, vox_occ_list, pred_occ_mask

    @force_fp32(apply_to=("x", "voxel_feat", "lss_depth"))
    def voxel_encoder(self, x, metas, instance_bank, training: bool, data=None, **kwargs):
        """
        Full voxel encoding pipeline:
        - view transform
        - voxel backbone+neck
        - temporal stream aggregation and permutation
        Args:
            instance_bank: Instance bank from queryagg (passed from duocc_head)
        Returns:
            voxel_feat_out: (B, W, L, H, C), lss_depth, vox_occ_list, pred_occ_mask
        """
        # Lightweight FPN: view transform + voxel backbone/neck encoding
        voxel_feat, lss_depth = self.lightweight_fpn(x, metas)
        
        # Forward to voxel stage which handles streamagg
        voxel_feat, vox_occ_list, pred_occ_mask = self.forward_voxel_stage(
            voxel_feat, metas, instance_bank, training, **kwargs
        )
        return voxel_feat, lss_depth, vox_occ_list, pred_occ_mask
      
    def forward(self,
                voxel_feat,  # [bs, embed_dims,D, H, W]
                prev_vox_feat=None,
                **kwargs,
                ) -> torch.Tensor:
        if prev_vox_feat is None:
            prev_vox_feat = voxel_feat.clone().detach()
        vox_occ_list = []
        query = None
        pred_occ_mask = None
        query = prev_vox_feat
        query, pred_occ_mask = self.temporal_conv_net(query) # [B,C,D,H,W]

        if self.training:
            if self.use_forecast_head:
                up_vox_occ = self.pred_up_block(query)
                vox_occ = self.forecast_occ_net(up_vox_occ)
                vox_occ_list.append(vox_occ.permute(0,1,4,3,2))
        if self.temp_cat_method == 'cat':
            query = self.cat_block(torch.cat([query,voxel_feat],dim=1))
        else:
            query = query + voxel_feat
        return query, vox_occ_list, pred_occ_mask

