import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils import normal_init, bias_init_with_prob, ConvModule

from basicsr.utils.registry import HEAD_REGISTRY

INF = 1e8

act_table = {
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'softmax': nn.Softmax
}

def build_act(opt):
    opt = deepcopy(opt)
    act_type = opt.pop('type')
    act =  act_table.get(act_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Act [{act.__class__.__name__}] is created.')
    return act

@HEAD_REGISTRY.register()
class ParamRecolorHead(nn.Module):

    def __init__(self,
                 param_channels,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 cate_down_pos=0,
                 cate_down_scale=2,
                 feat_down_scale=1,
                 param_nums = 16,
                 with_deform=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg = 'relu',
                 use_feature_mean=False,
                 feature_mean_pos=0,
                 last_param_layer_act_cfg=None):
        super(ParamRecolorHead, self).__init__()
        self.param_channels = param_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.cate_down_pos = cate_down_pos
        self.cate_down_scale = cate_down_scale
        self.feat_down_scale = feat_down_scale
        self.with_deform = with_deform
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.param_nums = param_nums
        self.param_channels = param_channels
        self.last_param_layer_act_cfg = last_param_layer_act_cfg
        self.use_feature_mean = use_feature_mean
        self.feature_mean_pos=feature_mean_pos
        self.act_cfg = act_cfg
        self._init_layers()

    def _init_layers(self):
        norm_cfg = self.norm_cfg
        self.param_convs = nn.ModuleList()
        if self.stacked_convs < 2:
            pool_layer_size = [8]
        elif self.stacked_convs < 3:
            pool_layer_size = [4,2]
        else:
            pool_layer_size = [4,2] + [2,]*(self.stacked_convs-2)
        up_size = 1
        for i in pool_layer_size:
            up_size *= i
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.seg_feat_channels//(self.feat_down_scale**(i-1))
            if not self.use_feature_mean:
                if i == 0:
                    self.param_convs.append(
                        nn.Upsample(size=up_size, mode='bilinear', align_corners=True)) 
                self.param_convs.append(
                    ConvModule(
                        chn,
                        self.seg_feat_channels//(self.feat_down_scale**i),
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        activation=self.act_cfg,
                        bias=norm_cfg is None))
                self.param_convs.append(
                    nn.AvgPool2d(pool_layer_size[i],pool_layer_size[i]))  
            else:
                if i >= self.feature_mean_pos:
                    self.param_convs.append(
                        ConvModule(
                            chn,
                            self.seg_feat_channels//(self.feat_down_scale**i),
                            1,
                            stride=1,
                            padding=0,
                            norm_cfg=norm_cfg,
                            activation=self.act_cfg,
                            bias=norm_cfg is None))
                else:
                    self.param_convs.append(
                        ConvModule(
                            chn,
                            self.seg_feat_channels//(self.feat_down_scale**i),
                            3,
                            stride=1,
                            padding=1,
                            norm_cfg=norm_cfg,
                            activation=self.act_cfg,
                            bias=norm_cfg is None))

        self.recolor_param = nn.Conv2d(
            self.seg_feat_channels//(self.feat_down_scale**i), self.param_nums*self.param_channels, 1)
        if self.last_param_layer_act_cfg is not None:
            self.recolor_param = [ self.recolor_param ]
            self.recolor_param.append( build_act(self.last_param_layer_act_cfg) ) 
            self.recolor_param = nn.Sequential(*self.recolor_param) 

    def init_weights(self):
        for m in self.param_convs:
            normal_init(m.conv, std=0.01)
        bias_param = bias_init_with_prob(0.01)
        normal_init(self.recolor_param, std=0.01, bias=bias_param)

    def concat_feats(self, feats):
        if not isinstance(feats, list) and (not isinstance(feats, tuple)):
            feats = [feats,]
        new_size = [feats[0].size(2)//self.cate_down_scale,feats[0].size(3)//self.cate_down_scale]
        feat_list = [F.interpolate(feats[0], scale_factor=1./self.cate_down_scale, mode='bilinear')]
        for i in range(1,len(feats)):
            feat_list.append( F.interpolate(feats[i], size=new_size, mode='bilinear') )
        return torch.cat(feat_list, dim=1)

    def mean_feats(self, feats):
        if not isinstance(feats, list) and (not isinstance(feats, tuple)):
            feats = [feats,]
        return torch.cat([torch.mean(feat, dim=(2,3), keepdim=True) for feat in feats], dim=1)

    def forward(self, feats):
        param_feat = self.concat_feats(feats)

        # cate branch
        for i, param_layer in enumerate(self.param_convs):
            if i == self.feature_mean_pos and self.use_feature_mean:
                param_feat = self.mean_feats(param_feat) 
            param_feat = param_layer(param_feat)

        param_pred = self.recolor_param(param_feat).view(param_feat.size(0),self.param_channels,1,-1)
        return param_pred

