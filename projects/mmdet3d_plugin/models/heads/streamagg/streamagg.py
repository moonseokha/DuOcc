# Copyright Seokha Moon. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.cnn import build_conv_layer, build_upsample_layer, build_norm_layer

from mmdet.models import build_backbone
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
                 ):
        super(StreamAgg, self).__init__()
        self.embed_dims = embed_dims
        self.use_forecast_head = use_forecast_head
        self.temp_cat_method = temp_cat_method
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

