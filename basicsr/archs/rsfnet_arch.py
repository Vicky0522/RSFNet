import torch.nn as nn
import torch
import math
from copy import deepcopy
import torch.nn.functional as F
import time

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils import get_root_logger
from basicsr.backbones import build_backbone
from basicsr.heads import build_head
from basicsr.necks import build_neck

from .renderer_arch import \
    Renderer_N16, Renderer_N1_K16, Renderer_N10, Renderer_N10_cascaded, Renderer_N1_K10

torch.autograd.set_detect_anomaly(True)

graph_table = {
    'Renderer_N16': Renderer_N16,
    'Renderer_N1_K16': Renderer_N1_K16,
    'Renderer_N10': Renderer_N10,
    'Renderer_N10_cascaded': Renderer_N10_cascaded,
    'Renderer_N1_K10': Renderer_N1_K10
}

def build_graph(opt):
    opt = deepcopy(opt)
    graph_type = opt.pop('type')
    net =  graph_table.get(graph_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Graph [{net.__class__.__name__}] is created.')
    return net

@ARCH_REGISTRY.register()
class RSFNet(nn.Module):

    def __init__(self,
                 backbone,
                 head,
                 graph={'type':'Renderer_N16'},
                 complementary_mask=False,
                 cache=True,
                 neck=None,
                 max_down_size=32,
                 init_weight=False,
                 mask_clustering=False,
                 consistent=False,
                 cluster_num=None,
                 pretrained=None):
        super(RSFNet, self).__init__()
        self.max_down_size = max_down_size
        self.backbone = build_backbone(backbone)
        self.with_neck = False
        if neck is not None:
            self.neck = build_neck(neck)
            self.with_neck = True
        self.head = build_head(head)
        self.graph = build_graph(graph)
        self.complementary_mask = complementary_mask
        self.cache = cache
        self.mask_clustering = mask_clustering
        self.consistent = consistent

        if init_weight:
            self.init_weights(pretrained)

        if self.mask_clustering:
            self.cluster_num = cluster_num if cluster_num is not None else 5

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained) 
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, img, params=None, masks=None, cache=None):
        begin = time.time()
        h, w = img.shape[2:]
        hh, ww = math.ceil(float(h)/self.max_down_size)*self.max_down_size, math.ceil(float(w)/self.max_down_size)*self.max_down_size
        pad_h = hh - h
        pad_w = ww - w
        if masks is not None:
            input_tensor = torch.cat([img, masks], dim=1)
        else:
            input_tensor = img
        x = F.pad(input_tensor, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2), "reflect")
        self.time_cost = [time.time() - begin]

        # for big input, but will introduce uncontiguous artifacts
        if max(hh, ww) > 720:
            s_ratio = 512. / max(hh, ww)
            x = F.interpolate(x, scale_factor=s_ratio, mode="bilinear")
        
        # extract feats
        x = self.extract_feat(x)
        self.time_cost.append( time.time() - begin )
        outs = list(self.head(x))
        self.time_cost += self.head.time_cost
        self.time_cost.append( time.time() - begin )

        if self.mask_clustering:
            results = self.graph.mask_clustering(outs[0], outs[1], self.cluster_num, self.consistent)
            #outs[0] = results.get('masks')
            #outs[1] = results.get('params')
            outs.append( results.get('masks') )
            print(results.get('params'))
            #print(results.get('params').shape)
            #print(results.get('params')[0,1:2,0,:])


        outs[0] = F.interpolate(outs[0], size=[hh,ww], mode="bilinear")#, align_corners=True)
        outs[0] = outs[0][:,:,pad_h//2:pad_h//2+h,pad_w//2:pad_w//2+w]
        #outs[0] = torch.zeros_like(outs[0]).to(outs[0].device)
        #outs[0][:,3:4,...] = 1.0

        if params is not None:
            outs[1] = params
        if masks is not None:
            outs[0] = outs[0] * masks
        if self.complementary_mask:
            outs[0] = torch.cat([outs[0], 1-torch.sum(outs[0],dim=1,keepdim=True).clamp(0,1)], dim=1)

        cache = self.cache if cache is None else cache
        
        if self.mask_clustering:
            # version 1
            #shift = outs[1][:,0:1,:,-4:-1].clone().permute(0,3,1,2)
            #params_tmp = outs[1].clone()
            #params_tmp[:,:,:,-4:-1] = 0.0
#            params_tmp[:,0:1,:,2] -= 1.0
            #results = self.graph.forward_cluster(img, outs[0], params_tmp, cache)
            #results['result'] = ((results['result'] + 1)/2. + shift - 0.5) / 0.5
            # version 2
            edit_p = torch.zeros_like(outs[1]).to(outs[1].device)[:,:-3,...]
            edit_p[:,1,...] += 1.0
            edit_p[:,5,...] += 0.2
            outs[2] = F.interpolate(outs[2], size=[hh,ww], mode="bilinear")
            alpha = torch.Tensor([1.0]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(outs[2].device)
            mask_c = torch.cat([self.graph.graph_cluster.gaussian_blur_use_conv2d(outs[2][:,m:m+1,...], alpha) for m in range(self.cluster_num)], axis=1) 
            mask_c = F.interpolate(mask_c[:,3:4,...], size=[hh,ww], mode="bilinear")
            mask_c = mask_c[:,:,pad_h//2:pad_h//2+h,pad_w//2:pad_w//2+w]
#            mask_c = torch.ones_like(mask_c).to(mask_c.device)
            delta_pix_masks = edit_p * mask_c#[:,3:4,...]
            results = self.graph(img, outs[0], outs[1], cache, delta_pix_masks)
            outs[0] = torch.cat([outs[0],F.interpolate(outs[2], size=[hh,ww], mode="bilinear")[:,:,pad_h//2:pad_h//2+h,pad_w//2:pad_w//2+w]], axis=1)
            
        else:
#            outs[1][:,0:1,:,1] += 0.2
#            outs[1][:,0:1,:,3] += 0.2
        #    outs[1][:,3:4,:,3] -= 0.5
         #   outs[1][:,3:4,:,3] -= 0.2
         #   outs[1][:,3:4,:,2] -= 0.2
            #outs[1][:,4:5,:,2] += 2.0
            #outs[1][:,4:5,:,7:10] += 0.04
            results = self.graph(img, outs[0], outs[1], cache)

        return_dict = {
            'result': results.get('result'),
            #'result': torch.zeros_like(img).to(img.device),
            'masks':  outs[0],
            'params': outs[1]
        }
        if cache:
            return_dict.update({'cache': results.get('cache_imgs')})

        self.time_cost.append( time.time() - begin )
            
        return return_dict

@ARCH_REGISTRY.register()
class RSFNet_MaskIn(nn.Module):

    def __init__(self,
                 backbone,
                 head,
                 graph={'type':'Renderer_N16'},
                 complementary_mask=False,
                 neck=None,
                 cache=True,
                 order=None,
                 max_down_size=32,
                 init_weight=False,
                 pretrained=None):
        super(RSFNet_MaskIn, self).__init__()
        self.max_down_size = max_down_size
        self.backbone = build_backbone(backbone)
        self.with_neck = False
        if neck is not None:
            self.neck = build_neck(neck)
            self.with_neck = True
        self.head = build_head(head)
        self.graph = build_graph(graph)
        self.complementary_mask = complementary_mask
        self.cache = cache
        self.order = order

        if init_weight:
            self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained) 
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, img, params=None, masks=None, cache=None):
        h, w = img.shape[2:]
        hh, ww = math.ceil(h/self.max_down_size)*self.max_down_size, math.ceil(w/self.max_down_size)*self.max_down_size
        pad_h = hh - h
        pad_w = ww - w
        if masks is not None:
            input_tensor = torch.cat([img, masks], dim=1)
        else:
            input_tensor = img
        x = F.pad(input_tensor, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2), "reflect")

        # for big input, but will introduce uncontiguous artifacts
        s_ratio = 360. / max(hh, ww)
        x = F.interpolate(x, scale_factor=s_ratio, mode="bilinear")

        if params is None:
            x = self.extract_feat(x)
            params = self.head(x)

        if masks is None:
            masks = torch.ones_like(img)[:,0:1,...].to(img.device)
        if self.complementary_mask:
            masks = torch.cat([masks, 1-torch.sum(masks,dim=1,keepdim=True).clamp(0,1)], dim=1)

        assert masks.size(1) == params.size(1), f"n_masks=={params.size(1)}, but input masks has {masks.size(1)} channels."

        cache = self.cache if cache is None else cache
        results = self.graph(img, masks, params, cache, self.order)

        return_dict = {
            'result': results.get('result'),
            'params': params
        }
        if cache:
            return_dict.update({'cache': results.get('cache_imgs')})
            
        return return_dict

