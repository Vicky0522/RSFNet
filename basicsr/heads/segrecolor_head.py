from copy import deepcopy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils import normal_init, bias_init_with_prob, ConvModule, get_root_logger

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

class QualifierMask(nn.Module):
    def __init__(self, qualifier_type="gauss", sigma=0.1, eps=1e-6):
        self.qualifier_type = qualifier_type
        self.eps = eps

        if qualifier_type == "gauss":
            self.beta = 2**0.5*3.1415926**2*sigma/8.
            self.alpha = 1./((2*3.1415926)**0.5*sigma)

    def forward(self, x, img):
        h, w = img.size()[-2:]
        img = [img[:,:,:h//2,:w//2],
               img[:,:,:h//2,w//2:],
               img[:,:,h//2:,:w//2],
               img[:,:,h//2:,w//2:]
              ]

        if self.qualifier_type == "gauss": 
            mean = [torch.sum(x[...,i:i+1,:]*img[i], dim=(2,3))/(torch.sum(x[...,i:i+1,:], dim=(2,3))+self.eps) for i in range(len(img))]
            std = [torch.sqrt(torch.sum(x[...,i:i+1,:]*(img[i]-mean[i])**2,dim=(2,3))/(torch.sum(x[...,i:i+1,:], dim=(2,3))+self.eps)) for i in range(len(img))]
            lower = [mean[i] - std[i] for i in range(len(img))]
            upper = [mean[i] + std[i] for i in range(len(img))]
            #sigma = [2*std[i] for 

        ### to be complemented 

            
        
            

@HEAD_REGISTRY.register()
class SegRecolorHead(nn.Module):

    def __init__(self,
                 num_segs,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 cate_down_pos=0,
                 cate_down_scale=2,
                 feat_down_scale=1,
                 mask_up_scale=[1,1,1,1],
                 param_nums = 16,
                 param_channels = None,
                 with_deform=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg = 'relu',
                 use_feature_mean=False,
                 last_param_layer_act_cfg=None,
                 last_layer_act_cfg=dict(type='sigmoid')):
        super(SegRecolorHead, self).__init__()
        self.num_segs = num_segs
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.cate_down_scale = cate_down_scale
        self.cate_down_pos = cate_down_pos
        self.feat_down_scale = feat_down_scale
        self.with_deform = with_deform
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.param_channels = param_channels if param_channels is not None else num_segs
        self.param_nums = param_nums
        self.last_layer_act_cfg = last_layer_act_cfg
        self.last_param_layer_act_cfg = last_param_layer_act_cfg
        self.use_feature_mean = use_feature_mean
        self.mask_up_scale = mask_up_scale[:stacked_convs]
        self.act_cfg = act_cfg

        self._init_layers()


    def _init_layers(self):
        norm_cfg = self.norm_cfg
        self.mask_convs = nn.ModuleList()
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
            if self.mask_up_scale[i] > 1:
                self.mask_convs.append(
                    nn.Upsample(scale_factor=self.mask_up_scale[i]))
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels//(self.feat_down_scale**i),
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    activation=self.act_cfg,
                    bias=norm_cfg is None))

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

        recolor_mask_list = [nn.Conv2d(self.seg_feat_channels//(self.feat_down_scale**i), self.num_segs, 1),]
        if self.last_layer_act_cfg is not None:
            recolor_mask_list.append( build_act(self.last_layer_act_cfg) )
        self.recolor_mask = nn.Sequential(*recolor_mask_list)
        self.recolor_param = nn.Conv2d(
            self.seg_feat_channels//(self.feat_down_scale**i), self.param_nums*self.param_channels, 1)
        if self.last_param_layer_act_cfg is not None:
            self.recolor_param = [ self.recolor_param ]
            self.recolor_param.append( build_act(self.last_param_layer_act_cfg) )
        #self.recolor_param = nn.Sequential(*self.recolor_param)

    def init_weights(self):
        for m in self.mask_convs:
            if isinstance(m, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        for m in self.param_convs:
            if isinstance(m, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        bias_ins = bias_init_with_prob(0.01)
        for m in self.recolor_mask:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01, bias=bias_ins)
        bias_param = bias_init_with_prob(0.01)
        normal_init(self.recolor_param, std=0.01, bias=bias_param)

    def concat_feats(self, feats):
        new_size = [feats[0].size(2)//self.cate_down_scale,feats[0].size(3)//self.cate_down_scale]
        feat_list = [F.interpolate(feats[0], scale_factor=1./self.cate_down_scale, mode='bilinear')]
        #feat_list = [feats[0],]
        for i in range(1,len(feats)):
            feat_list.append( F.interpolate(feats[i], size=new_size, mode='bilinear') )
        return torch.cat(feat_list, dim=1)

    def mean_feats(self, feats):
        return torch.cat([torch.mean(feat, dim=(2,3), keepdim=True) for feat in feats], dim=1)
   
    def forward(self, feats):
        begin = time.time()
        ins_feat = self.concat_feats(feats)
        self.time_cost = [time.time()-begin]
         
        if self.use_feature_mean:
            param_feat = self.mean_feats(feats)
        else:
            param_feat = ins_feat.clone()
        self.time_cost.append( time.time()-begin )
        # ins branch
        # concat coord

        for i, mask_layer in enumerate(self.mask_convs):
            ins_feat = mask_layer(ins_feat)
        self.time_cost.append( time.time()-begin )

        mask_feat = F.interpolate(ins_feat, scale_factor=self.cate_down_scale, mode='bilinear')
        mask_pred = self.recolor_mask(mask_feat)
        self.time_cost.append( time.time()-begin )
        #mask_max = torch.max(torch.max(mask_pred,dim=3,keepdim=True)[0],dim=2,keepdim=True)[0] + 1e-8
        #mask_pred = mask_pred / mask_max

        # cate branch
        for i, param_layer in enumerate(self.param_convs):
            param_feat = param_layer(param_feat)

        param_pred = self.recolor_param(param_feat).view(param_feat.size(0),self.param_channels,1,-1)
        self.time_cost.append( time.time()-begin )

        return mask_pred, param_pred

