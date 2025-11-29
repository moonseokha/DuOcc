# Copyright Seokha Moon. All rights reserved.
import torch
import torch.nn as nn
from typing import Optional

from ...modules.utils import init_weights
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils import softmax

from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.core import multi_apply  # (kept for config compatibility)

__all__ = ["DQA"]

# Box parameter indices
X, Y, Z, W, L, H, SIN, COS = 0, 1, 2, 3, 4, 5, 6, 7


def box3d_to_corners(box3d: torch.Tensor) -> torch.Tensor:
    """Convert 3D boxes (cx, cy, cz, log(w,l,h), sin, cos) to 8 corners.

    Args:
        box3d: [..., 8] tensor

    Returns:
        corners: [..., 8, 3]
    """
    boxes = box3d.clone().detach()
    # width/length/height are stored in log-space
    boxes[..., 3:6] = boxes[..., 3:6].exp()

    # 8 normalized cube corners in a consistent order
    corners_norm = torch.stack(
        torch.meshgrid(
            torch.arange(2), torch.arange(2), torch.arange(2)
        ),
        dim=-1,
    ).view(-1, 3)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]] - 0.5  # [8, 3]

    # scale by box size
    corners = (
        boxes[..., None, [W, L, H]]
        * corners_norm.to(boxes.device).reshape(1, 8, 3)
    )  # [..., 8, 3]

    # rotation around z
    rot_mat = torch.eye(3, device=boxes.device).view(1, 1, 3, 3)
    rot_mat = rot_mat.repeat(boxes.shape[0], boxes.shape[1], 1, 1)
    rot_mat[..., 0, 0] = boxes[..., COS]
    rot_mat[..., 0, 1] = -boxes[..., SIN]
    rot_mat[..., 1, 0] = boxes[..., SIN]
    rot_mat[..., 1, 1] = boxes[..., COS]

    corners = (rot_mat.unsqueeze(2) @ corners.unsqueeze(-1)).squeeze(-1)
    return corners + boxes[..., None, :3]


@PLUGIN_LAYERS.register_module()
class DQA(nn.Module):
    """Dynamic Query Aggregation (DQA).

    - Takes query features (agent_feat) and voxel features (vox_feat)
    - Builds query-to-voxel edges
    - Applies graph-based cross-attention
    - Optionally predicts occupancy logits
    """

    def __init__(
        self,
        embed_dims: int,
        conv_cfg: dict = None,
        grid_config: dict = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        conv3d_layers: int = 1,
        num_classes: int = 18,
        down_ratio: int = 4,
        use_edge_pos: bool = False,
        num_cams: int = 6,
        num_levels: int = 4,
        num_points: int = 6,
        num_groups: int = 8,
        query_to_vox: bool = True,
        without_occ: bool = False,
        ffn: dict = None,
        no_norm: bool = False,
        surround_occ: bool = False,
    ) -> None:
        super(DQA, self).__init__()

        self.embed_dims = embed_dims
        self.conv3d_layers = conv3d_layers
        self.num_classes = num_classes
        self.down_ratio = down_ratio

        self.use_edge_pos = use_edge_pos
        self.without_occ = without_occ
        self.query_to_vox = query_to_vox
        self.surround_occ = surround_occ
        self.no_norm = no_norm

        # Voxel grid meta
        self.vox_size = [
            int((grid_config["y"][1] - grid_config["y"][0]) / grid_config["y"][-1]),
            int((grid_config["x"][1] - grid_config["x"][0]) / grid_config["x"][-1]),
            int((grid_config["z"][1] - grid_config["z"][0]) / grid_config["z"][-1]),
        ]
        self.vox_res = [
            grid_config["y"][-1],
            grid_config["x"][-1],
            grid_config["z"][-1],
        ]
        self.vox_lower_point = [
            grid_config["y"][0],
            grid_config["x"][0],
            grid_config["z"][0],
        ]

        if not self.no_norm:
            self.norm_cross = nn.LayerNorm(embed_dims)

        # Cross-attention from query → voxel
        if self.query_to_vox:
            self.cross_attn_Q2V = CrossAttentionLayer(
                embed_dim=embed_dims,
                num_heads=num_heads,
                dropout=dropout,
                use_edge_pos=use_edge_pos,
                no_norm=no_norm,
            )

        # Occupancy head
        out_dims = embed_dims // down_ratio
        if not self.without_occ:
            self.up_block = nn.Sequential(
                nn.ConvTranspose3d(
                    embed_dims,
                    out_dims,
                    kernel_size=(2, 2, 2),
                    stride=(2, 2, 2),
                ),
                nn.BatchNorm3d(out_dims),
                nn.ReLU(inplace=True),
            )
            self.proj_layer = ConvModule(
                out_dims,
                embed_dims,
                kernel_size=1,
                stride=1,
                bias=True,
                conv_cfg=dict(type="Conv3d"),
            )
            self.vox_occ_net = nn.Conv3d(embed_dims, num_classes, 1, 1, 0)

        self.apply(init_weights)

    def forward(
        self,
        agent_feat: torch.Tensor = None,  # [B, N, C]
        vox_feat: torch.Tensor = None,  # [B, H, W, D, C]
        agent_pos: torch.Tensor = None,  # [B, N, 11]
        indices=None,
        metas=None,
        voxel_pos=None,
    ) -> torch.Tensor:
        """Forward.

        Returns:
            vox_with_query: [B, H, W, D, C]
            vox_occ:        [B, num_classes, H_up, W_up, D_up] or None
            up_vox_occ:     [B, C, H_up, W_up, D_up] or None
        """
        B, H, W, D, _ = vox_feat.shape
        vox_flatten_size = H * W * D

        vox_occ = None
        up_vox_occ = None

        # 1) Query-to-voxel cross attention
        if self.query_to_vox:
            vox_with_query = self._apply_query_to_voxel_attention(
                agent_feat=agent_feat,
                vox_feat=vox_feat,
                agent_pos=agent_pos,
                indices=indices,
                metas=metas,
                voxel_pos=voxel_pos,
                B=B,
                vox_flatten_size=vox_flatten_size,
            )
        else:
            # Just reshape to [B, C, H, W, D]
            vox_with_query = vox_feat.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3)

        # 2) Occupancy prediction head
        if not self.without_occ:
            up_vox_occ = self.up_block(vox_with_query)           # [B, C_out, H_up, W_up, D_up]
            up_vox_occ = self.proj_layer(up_vox_occ)             # [B, C, H_up, W_up, D_up]
            vox_occ = self.vox_occ_net(up_vox_occ)               # [B, num_classes, H_up, W_up, D_up]
            vox_occ = vox_occ.permute(0, 1, 3, 2, 4)             # [B, num_classes, W_up, H_up, D_up]

        # 3) Return voxel features back to [B, H, W, D, C]
        vox_with_query = vox_with_query.permute(0, 2, 3, 4, 1)

        return vox_with_query, vox_occ, up_vox_occ

    def _apply_query_to_voxel_attention(
        self,
        agent_feat: torch.Tensor,
        vox_feat: torch.Tensor,
        agent_pos: torch.Tensor,
        indices,
        metas,
        voxel_pos: Optional[torch.Tensor],
        B: int,
        vox_flatten_size: int,
    ) -> torch.Tensor:
        """Build query→voxel edges and run CrossAttentionLayer."""
        # Common constants as tensors (created only once per forward)
        vox_lower = torch.tensor(
            self.vox_lower_point,
            dtype=agent_pos.dtype,
            device=agent_pos.device,
        )
        vox_res = torch.tensor(
            self.vox_res,
            dtype=agent_pos.dtype,
            device=agent_pos.device,
        )

        vox_index_tensors = []    # voxel indices (flattened) affected by each query
        query_index_tensors = []  # query indices corresponding to each voxel index
        agent_pos_tensors = []    # relative agent positions used as edge attributes

        query_total = 0

        for b in range(B):
            # Queries used in the current batch element
            batch_agent_pos = agent_pos[b][indices[b]]  # [N_q, 11]

            # (1) 3D box corners & center voxel position
            bottom_pos = box3d_to_corners(
                batch_agent_pos.unsqueeze(0)
            ).squeeze(0)  # [N_q, 8, 3]

            center_pos = (batch_agent_pos[:, :3] - vox_lower) / vox_res  # [N_q, 3]

            vox_pos = ((bottom_pos - vox_lower) / vox_res).round()  # [N_q, 8, 3]

            # SurroundOcc coordinate transformation
            if self.surround_occ:
                lidar2ego_inv = torch.as_tensor(
                    metas["img_metas"][b]["lidar2ego_inv"],
                    device=vox_pos.device,
                    dtype=vox_pos.dtype,
                )  # [4, 4]
                ones = torch.ones_like(vox_pos[..., :1])  # [N_q, 8, 1]
                vox_pos_h = torch.cat([vox_pos, ones], dim=-1)  # [N_q, 8, 4]
                vox_pos_h = torch.einsum("ij,...j->...i", lidar2ego_inv, vox_pos_h)
                vox_pos = vox_pos_h[..., :3]  # [N_q, 8, 3]

            # (2) Compute voxel bbox (min/max) for each query & clamp to grid
            min_pos = vox_pos.min(dim=1)[0]
            max_pos = vox_pos.max(dim=1)[0]

            min_pos[:, 0] = torch.clamp(min_pos[:, 0], 0, self.vox_size[0] - 1)
            min_pos[:, 1] = torch.clamp(min_pos[:, 1], 0, self.vox_size[1] - 1)
            min_pos[:, 2] = torch.clamp(min_pos[:, 2], 0, self.vox_size[2] - 1)
            max_pos[:, 0] = torch.clamp(max_pos[:, 0], 0, self.vox_size[0] - 1)
            max_pos[:, 1] = torch.clamp(max_pos[:, 1], 0, self.vox_size[1] - 1)
            max_pos[:, 2] = torch.clamp(max_pos[:, 2], 0, self.vox_size[2] - 1)

            min_pos_int = min_pos.to(dtype=torch.long)
            max_pos_int = max_pos.to(dtype=torch.long)

            # (3) Generate voxel indices inside the box for each query
            for num in range(vox_pos.shape[0]):
                xs = torch.arange(
                    min_pos_int[num, 0],
                    max_pos_int[num, 0] + 1,
                    device=vox_feat.device,
                )
                ys = torch.arange(
                    min_pos_int[num, 1],
                    max_pos_int[num, 1] + 1,
                    device=vox_feat.device,
                )
                zs = torch.arange(
                    min_pos_int[num, 2],
                    max_pos_int[num, 2] + 1,
                    device=vox_feat.device,
                )

                mesh_index = torch.stack(
                    torch.meshgrid(xs, ys, zs),
                    dim=-1,
                ).reshape(-1, 3)  # [n_i, 3]
                mesh_index = torch.unique(mesh_index, dim=0)  # remove duplicates

                # Compute relative distance & edge_attr (agent position)
                relative_dist = mesh_index.to(center_pos.dtype) - center_pos[num]  # [n_i, 3]

                base_agent = batch_agent_pos[num].unsqueeze(0).expand(
                    mesh_index.shape[0], -1
                )  # [n_i, 11]
                base_agent = base_agent.clone()
                base_agent[:, :3] = relative_dist  # replace xyz with relative coordinates
                relative_agent_pos = base_agent  # [n_i, 11]

                # (4) Compute flattened voxel index
                idx = (
                    mesh_index[:, 1] * self.vox_size[1] * self.vox_size[2]
                    + mesh_index[:, 0] * self.vox_size[2]
                    + mesh_index[:, 2]
                )
                idx = torch.clamp(
                    idx,
                    min=0,
                    max=self.vox_size[0]
                    * self.vox_size[1]
                    * self.vox_size[2]
                    - 1,
                )
                idx = idx + b * vox_flatten_size  # batch offset

                # (5) Accumulate indices and edge attributes
                vox_index_tensors.append(idx)               # (n_i,)
                agent_pos_tensors.append(relative_agent_pos) # (n_i, 11)
                query_index_tensors.append(
                    torch.full(
                        (idx.shape[0],),
                        query_total,
                        dtype=torch.long,
                        device=vox_feat.device,
                    )
                )
                query_total += 1

        # (6) Concatenate all tensors once
        if len(vox_index_tensors) > 0:
            vox_indices = torch.cat(vox_index_tensors, dim=0)         # [E]
            query_indices = torch.cat(query_index_tensors, dim=0)     # [E]
            selected_agent_pos = torch.cat(agent_pos_tensors, dim=0)  # [E, 11]
        else:
            # Fallback to avoid failures in degenerate cases (no edges)
            vox_indices = torch.zeros(1, dtype=torch.long, device=vox_feat.device)
            query_indices = torch.zeros(1, dtype=torch.long, device=vox_feat.device)
            selected_agent_pos = torch.zeros(
                1, 11,
                dtype=agent_pos.dtype,
                device=vox_feat.device,
            )

        edge_index_Q2V = torch.stack([query_indices, vox_indices], dim=0)  # [2, E]

        # (7) Gather query features: [sum_N, C]
        selected_agent_feat = torch.cat(
            [agent_feat[b][indices[b]] for b in range(agent_feat.shape[0])],
            dim=0,
        )

        # (8) Cross-attention 실행
        vox_feat_flat = vox_feat.reshape(-1, self.embed_dims)
        if self.use_edge_pos:
            vox_with_query = self.cross_attn_Q2V(
                x=(selected_agent_feat, vox_feat_flat),
                edge_index=edge_index_Q2V,
                edge_attr=selected_agent_pos,
                voxel_pos_embedding=voxel_pos,
            )
        else:
            vox_with_query = self.cross_attn_Q2V(
                x=(selected_agent_feat, vox_feat_flat),
                edge_index=edge_index_Q2V,
                voxel_pos_embedding=voxel_pos,
            )

        if not self.no_norm:
            vox_with_query = self.norm_cross(vox_with_query)

        # [B, H, W, D, C] → [B, C, H, W, D]
        vox_with_query = vox_with_query.reshape(
            -1,
            self.vox_size[0],
            self.vox_size[1],
            self.vox_size[2],
            self.embed_dims,
        )
        vox_with_query = vox_with_query.permute(0, 4, 1, 2, 3)
        return vox_with_query

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

class CrossAttentionLayer(MessagePassing):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_edge_pos: bool = False,
                 no_norm:bool = False,
                 **kwargs) -> None:
        super(CrossAttentionLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_edge_pos = use_edge_pos
        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        self.no_norm = no_norm
        if self.use_edge_pos:
            self.lin_k_edge = nn.Linear(11, embed_dim)
            self.lin_v_edge = nn.Linear(11, embed_dim)


        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        if no_norm == False:
            self.norm1 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.alpha = 0
        self.index = 0

    def forward(self,
                x: torch.Tensor,
                edge_index: Adj,
                edge_attr: OptTensor = None,
                voxel_pos_embedding: OptTensor = None,
                size: Size = None,) -> torch.Tensor:
        x_source,x_target = x
        if self.no_norm:
            x_target = x_target + self._mha_block(x_target,x_source, edge_index,edge_attr,voxel_pos_embedding, size)
        else:
            x_target = x_target + self._mha_block(x_target,self.norm1(x_source), edge_index,edge_attr,voxel_pos_embedding, size)
        x_target = x_target + self._ff_block(self.norm3(x_target))
        return x_target

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: OptTensor,
                vox_pos_embedding_i: OptTensor,
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        if vox_pos_embedding_i is not None:
            x_i = x_i + vox_pos_embedding_i
        query = self.lin_q_node(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_node = self.lin_k_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_node = self.lin_v_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        if self.use_edge_pos:
            key_edge = self.lin_k_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
            value_edge = self.lin_v_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5

        if self.use_edge_pos:
            alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        else:
            alpha = (query * key_node).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        self.alpha = alpha
        self.index = index
        alpha = self.attn_drop(alpha)
        if self.use_edge_pos:
            return (value_node + value_edge) * alpha.unsqueeze(-1)* x_j.sum(-1).bool()[...,None,None]
        return value_node * alpha.unsqueeze(-1)* x_j.sum(-1).bool()[...,None,None]

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        return inputs * gate

    def _mha_block(self,
                   x_target: torch.Tensor,
                   x_source: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: OptTensor,
                   vox_pos_embedding: OptTensor,
                   size: Size) -> torch.Tensor:
        if vox_pos_embedding is not None:
            vox_pos_embedding = vox_pos_embedding.flatten(0,1)
        x = self.out_proj(self.propagate(edge_index=edge_index, x=(x_source,x_target), edge_attr=edge_attr,vox_pos_embedding=vox_pos_embedding,size=size))
        return self.proj_drop(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)