import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils import normal_init, bias_init_with_prob, ConvModule

from basicsr.utils.registry import HEAD_REGISTRY


INF = 1e8

@HEAD_REGISTRY.register()
class MaskHead(nn.Module):

    def __init__(self,
                 num_segs,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 cate_down_pos=0,
                 with_deform=False,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SegRecolorHead, self).__init__()
        self.num_segs = num_segs
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.cate_down_pos = cate_down_pos
        self.with_deform = with_deform
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.mask_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        self.recolor_mask = nn.Sequential(
            [nn.Conv2d(self.seg_feat_channels, self.num_segs, 1),
             nn.Softmax(dim=1)
             ])

    def init_weights(self):
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)
        bias_ins = bias_init_with_prob(0.01)
        normal_init(self.recolor_mask, std=0.01, bias=bias_ins)

    def concat_feats(self, feats):
        new_size = [feats[0].size(2)//2,feats[0].size(3)//2]
        return torch.cat(
                [F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'), 
                 F.interpolate(feats[1], size=new_size, mode='bilinear'),
                 F.interpolate(feats[2], size=new_size, mode='bilinear'), 
                 F.interpolate(feats[3], size=new_size, mode='bilinear'),
                 F.interpolate(feats[4], size=new_size, mode='bilinear')], dim=1)

    def forward(self, feats):
        feats = self.concat_feats(feats)

        ins_feat = feats
        for i, mask_layer in enumerate(self.mask_convs):
            ins_feat = mask_layer(ins_feat)

        mask_feat = F.interpolate(ins_feat, scale_factor=2, mode='bilinear')
        mask_pred = self.recolor_mask(mask_feat)

        return mask_pred

