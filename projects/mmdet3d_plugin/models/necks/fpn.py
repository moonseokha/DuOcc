# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Seokha Moon
# ---------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import NECKS
import torch
from mmcv.cnn import ConvModule, xavier_init

def swish(x):
    return x * x.sigmoid()


class LayerCombineModule(nn.Module):
    def __init__(self, num_input=2):
        super().__init__()
        self.weights = nn.Parameter(
            torch.ones(num_input, dtype=torch.float32).view(1, 1, 1, 1, -1),
            requires_grad=True
        )

    def forward(self, inputs):

        weights = self.weights.relu()
        norm_weights = weights / (weights.sum() + 0.0001)

        out = (norm_weights*torch.stack(inputs, dim=-1)).sum(dim=-1)
        return swish(out)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x 

class SingleBiFPN(nn.Module):
    def __init__(self, in_channels, out_channels, no_norm_on_lateral=True, conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 last_fpn = False):
        super().__init__()

        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg

        self.lateral_convs = nn.ModuleList()
        self.lateral_combine = nn.ModuleList()
        self.lateral_combine_conv = nn.ModuleList()
        self.out_combine = nn.ModuleList()
        self.out_combine_conv = nn.ModuleList()
        self.last_fpn = last_fpn
        for i, in_channel in enumerate(in_channels):
            if in_channel != out_channels:
                self.lateral_convs.append(ConvModule(
                    in_channel,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False))
            else:
                self.lateral_convs.append(Identity())
            if i != len(in_channels)-1:
                self.lateral_combine.append(LayerCombineModule(2))
                self.lateral_combine_conv.append(ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=None,
                    inplace=False)
                )
            if self.last_fpn == False:
                if i != 0:
                    self.out_combine.append(LayerCombineModule(
                        3 if i != len(in_channels)-1 else 2))
                    self.out_combine_conv.append(ConvModule(
                        out_channels,
                        out_channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                        act_cfg=None,
                        inplace=False))

    def forward(self, inputs):

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals = laterals + \
            inputs[len(self.lateral_convs):]  # p3,p4,p5,p6,p7

        # top to down
        outs = [laterals[i] for i in range(len(laterals))]

        for i in range(len(laterals)-1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.

            if 'scale_factor' in self.upsample_cfg:
                up_feat = F.interpolate(outs[i],
                                        **self.upsample_cfg)
            else:
                prev_shape = outs[i-1].shape[2:]
                up_feat = F.interpolate(
                    outs[i], size=prev_shape, **self.upsample_cfg)
            # weight combine
            outs[i-1] = self.lateral_combine_conv[i -
                                                  1](self.lateral_combine[i-1]([outs[i-1], up_feat]))
        if self.last_fpn:
            return outs
        # down to top
        for i in range(len(outs)-1):
            down_feat = F.max_pool2d(outs[i], 3, stride=2, padding=1)
            cur_outs = outs[i+1]
            if i != len(laterals)-2:
                cur_inputs = laterals[i+1]
                outs[i +
                     1] = self.out_combine[i]([down_feat, cur_outs, cur_inputs])
            else:
                outs[i+1] = self.out_combine[i]([down_feat, cur_outs])
            outs[i+1] = self.out_combine_conv[i](outs[i+1])

        return outs


@NECKS.register_module()
class BiFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=160,
                 num_outs=4,
                 start_level=0,
                 end_level=-1,
                 num_repeat=3,
                 num_repeat_detach= None,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 use_deformable_func=True,):
        super(BiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.num_repeat = num_repeat
        self.num_repeat_detach = num_repeat_detach
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.use_deformable_func=use_deformable_func
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.downsample_convs = nn.ModuleList()
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_conv = nn.Sequential(
                    ConvModule(
                    in_channels,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False),
                    nn.MaxPool2d(3,2,1)
                    )
                self.downsample_convs.append(extra_conv)

        out_channels = out_channels if self.add_extra_convs else self.in_channels[
            self.backbone_end_level-1]
        self.bi_fpn = nn.ModuleList()
        if self.num_repeat_detach is not None:
            self.num_repeat += self.num_repeat_detach
        for i in range(self.num_repeat):
            if i == 0:
                in_channels = self.in_channels[self.start_level:self.backbone_end_level]+[
                    out_channels]*extra_levels
            else:
                in_channels = [self.out_channels]*num_outs
            last_fpn=False
            if not self.use_deformable_func:
                if i == (self.num_repeat-1):
                    last_fpn = True
            self.bi_fpn.append(SingleBiFPN(in_channels, self.out_channels, no_norm_on_lateral=no_norm_on_lateral,
                                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, upsample_cfg=upsample_cfg,last_fpn=last_fpn))
            

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function"""
        assert len(inputs) == len(self.in_channels)
        # build laterals
        outs = list(inputs[self.start_level:self.backbone_end_level])
        used_backbone_levels = len(outs)
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 3, stride=2, padding=1))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                for i in range(self.num_outs-used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.downsample_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.downsample_convs[i](outs[-1]))

        # p2,p3,p4,p5,p6,p7
        # forward to bifpn
        for i in range(self.num_repeat):
            outs = self.bi_fpn[i](outs)
        return outs



@NECKS.register_module()
class CustomFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 out_ids=[],
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 use_DETR=False,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CustomFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.use_DETR = use_DETR
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.out_ids = out_ids
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            if i in self.out_ids:
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        laterals_copy = None
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterrals_copy = None
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
                
        if self.use_DETR:
            laterals_copy = [lateral.clone() for lateral in laterals]
                
        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in self.out_ids]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        if self.use_DETR:
            return outs[0], laterals_copy
        return outs[0], laterals_copy



@NECKS.register_module()
class MultiFPN(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 mid_level = 2,
                 out_ids=[],
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 use_DETR=False,
                 detach=False,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(MultiFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.mid_level = mid_level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.detach = detach
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.out_ids = out_ids
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.mid_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
        for i in range(self.start_level, len(out_ids)):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)
            
    @auto_fp16()
    def forward(self, first_features, inter_features):
        # """Forward function."""
        # assert len(inputs) == len(self.in_channels)

        # build laterals
        if self.detach:
            first_features = [feature.clone().detach() for feature in first_features]
            inter_features = [feature.clone().detach() for feature in inter_features]
            #     first_features[num] = first_features[num].clone().detach()
            # for num in range(len(inter_features)):
            #     inter_features[num] = inter_features[num].clone().detach()
        laterals = [
            lateral_conv(first_features[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)+1
        laterals.extend(inter_features)
        # build top-down path\
        for i in range(used_backbone_levels - 1, 0, -1):
            upsampled = F.interpolate(laterals[i].clone(), **self.upsample_cfg) if 'scale_factor' in self.upsample_cfg else \
                        F.interpolate(laterals[i].clone(), size=laterals[i - 1].shape[2:], **self.upsample_cfg)
            laterals[i - 1] = laterals[i - 1] + upsampled
        
        outs = [self.fpn_convs[i](laterals[i]) for i in self.out_ids]
        return outs