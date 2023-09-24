import math

import torch
from torch import nn as nn
from torch.nn import functional as F
from kmeans_pytorch import kmeans, kmeans_predict

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer
from basicsr.ops.color import rgb2lab_func

@ARCH_REGISTRY.register()
class Renderer_N16(nn.Module):
    """
    Basic Graph for Recolor.
    """

    def __init__(self, kernel_size=51, min_sigma=3, eps=1e-6):
        super(Renderer_N16, self).__init__()
        x_range = torch.linspace(-kernel_size//2,kernel_size-kernel_size//2, kernel_size)
        y_range = torch.linspace(-kernel_size//2,kernel_size-kernel_size//2, kernel_size)
        x, y = torch.meshgrid(x_range, y_range)
        self.x = nn.Parameter(x, requires_grad=False)
        self.y = nn.Parameter(y, requires_grad=False)
        self.eps = eps
        self.min_sigma = min_sigma
        self.kernel_size = kernel_size

        self.pad = nn.ReflectionPad2d((kernel_size-1)//2)

    def render(self, image, mask,
               contrast, saturation, hue, temp,
               lift, gamma, gain, shift):
        
        # contrast [0,...)
        contrast = contrast + 1
        delta = contrast * image + (1 - contrast) * torch.mean(image) - image

        # saturation [0,1]
        saturation = (saturation + 1) / 2.
        image_L = rgb2lab_func(image)[:,0:1,...]/100.0
        delta += 2 * saturation * image + (1 - 2 * saturation) * image_L - image

        # hue [0,1] 
        hue = (hue + 1) / 2.
        delta[:,0:1,...] += image[:,0:1,...] * (hue - 0.5) / 0.5 * 0.1
        delta[:,2:3,...] += image[:,2:3,...] * (hue - 0.5) / 0.5 * 0.1
        delta[:,1:2,...] += image[:,1:2,...] * (0.5 - hue) / 0.5 * 0.05

        # temp [0,1]
        temp = (temp + 1) / 2.
        delta[:,0:1,...] += image[:,0:1,...] * F.leaky_relu( 
            (temp - 0.5) / 0.5 * (253 - 128) / 128,
            (128 - 104) / (253 - 128)
        )
        delta[:,1:2,...] += image[:,1:2,...] * F.leaky_relu( 
            (temp - 0.5) / 0.5 * (99 - 128) / 128,
            0.0
        )
        delta[:,2:3,...] += image[:,2:3,...] * F.leaky_relu( 
            (temp - 0.5) / 0.5 * (16 - 128) / 128,
            (128 - 180) / (16 - 128)
        )

        # lift
        delta[:,0:1,...] += 2*lift[...,0:1]*(1 - image[:,0:1,...])
        delta[:,1:2,...] += 2*lift[...,1:2]*(1 - image[:,1:2,...])
        delta[:,2:3,...] += 2*lift[...,2:3]*(1 - image[:,2:3,...])

        # gain
        delta[:,0:1,...] += 2*gain[...,0:1]*image[:,0:1,...]
        delta[:,1:2,...] += 2*gain[...,1:2]*image[:,1:2,...]
        delta[:,2:3,...] += 2*gain[...,2:3]*image[:,2:3,...]

        # gamma
        delta[:,0:1,...] += gamma[...,0:1]*7.2*(0.25-(image[:,0:1,...]-0.5)**2)
        delta[:,1:2,...] += gamma[...,1:2]*7.2*(0.25-(image[:,1:2,...]-0.5)**2)
        delta[:,2:3,...] += gamma[...,2:3]*7.2*(0.25-(image[:,2:3,...]-0.5)**2)

        # shift
        delta[:,0:1,...] += shift[...,0:1]
        delta[:,1:2,...] += shift[...,1:2]
        delta[:,2:3,...] += shift[...,2:3]

        return delta * mask 

    def gaussian_blur_use_conv2d(self, mask, alpha):
        y = self.y.expand([mask.shape[0], 1, -1, -1])
        x = self.x.expand([mask.shape[0], 1, -1, -1])
        sigma = torch.clamp(alpha,0,1)*(self.kernel_size-self.min_sigma)+self.min_sigma
        kernel = 1./(2*3.1415926*sigma**2) * torch.exp(-(x**2 + y**2)/(2*3.1415926*sigma**2))
        kernel /= torch.sum(kernel,dim=(1,2,3),keepdims=True)
        output = torch.zeros_like(mask).to(mask.device)

        return F.conv2d(self.pad(mask.permute(1,0,2,3)),weight=kernel,groups=kernel.size(0)).permute(1,0,2,3)

    def forward(self, input, masks, params, cache=False, *args, **kwargs):
        assert masks.size(1) == params.size(1), f"n_masks=={masks.size(1)}, but params has {params.size(1)} channels."
        concat_graph = []
        for m in range(masks.size(1)):
            contrast = params[:,m:m+1,:,0:1]
            saturation = params[:,m:m+1,:,1:2]
            hue = params[:,m:m+1,:,2:3]
            temp = params[:,m:m+1,:,3:4]
            lift = params[:,m:m+1,:,4:7]
            gamma = params[:,m:m+1,:,7:10]
            gain = params[:,m:m+1,:,10:13]
            shift = params[:,m:m+1,:,13:16]
            alpha = params[:,m:m+1,:,16:17]

            mask = self.gaussian_blur_use_conv2d(masks[:,m:m+1,...], alpha) 

            rendered_image_delta = self.render((input+1)/2., mask, contrast, saturation, hue, temp, lift, gamma, gain, shift)
            concat_graph.append(rendered_image_delta)

        return_dict = {
            "result": (sum(concat_graph) + (input+1)/2. - 0.5)/0.5
        }
        if cache:
            rendered_image = torch.cat([((delta + (input+1)/2.)-0.5)/0.5 for delta in concat_graph],dim=1)
            return_dict.update({'cache_imgs': rendered_image})
        return return_dict

class Renderer_N1_K16(nn.Module):
    """
    Basic Graph for Recolor.
    """

    def __init__(self, eps=1e-15):
        super(Renderer_N1_K16, self).__init__()
        self.eps = eps
        self.MaskCluster = MaskCluster()
        self.graph_cluster = Renderer_N16()

    def render(self, image, 
               contrast, saturation, hue, temp,
               lift, gamma, gain, shift):
        
        # contrast [0,...)
        contrast = contrast + 1
        delta = contrast * image + (1 - contrast) * torch.mean(image) - image

        # saturation [0,1]
        saturation = (saturation + 1) / 2.
        image_L = rgb2lab_func(image)[:,0:1,...]/100.0
        delta += 2 * saturation * image + (1 - 2 * saturation) * image_L - image

        # hue [0,1] 
        hue = (hue + 1) / 2.
        delta[:,0:1,...] += image[:,0:1,...] * (hue - 0.5) / 0.5 * 0.1
        delta[:,2:3,...] += image[:,2:3,...] * (hue - 0.5) / 0.5 * 0.1
        delta[:,1:2,...] += image[:,1:2,...] * (0.5 - hue) / 0.5 * 0.05

        # temp [0,1]
        temp = (temp + 1) / 2.
        delta[:,0:1,...] += image[:,0:1,...] * F.leaky_relu( 
            (temp - 0.5) / 0.5 * (253 - 128) / 128,
            (128 - 104) / (253 - 128)
        )
        delta[:,1:2,...] += image[:,1:2,...] * F.leaky_relu( 
            (temp - 0.5) / 0.5 * (99 - 128) / 128,
            0.0
        )
        delta[:,2:3,...] += image[:,2:3,...] * F.leaky_relu( 
            (temp - 0.5) / 0.5 * (16 - 128) / 128,
            (128 - 180) / (16 - 128)
        )

        # lift
        delta += 2*lift*(1 - image)

        # gain
        delta += 2*gain*image

        # gamma
        delta += gamma*7.2*(0.25-(image-0.5)**2)

        # shift
        delta += shift

        return delta 

    def forward(self, input, masks, params, cache=False, *args, **kwargs):
        assert masks.size(1) == params.size(1) - 3, f"n_masks=={masks.size(1)}, but params has {params.size(1)} channels."
        pix_masks = masks * params[:,:-3,...]
        contrast = pix_masks[:,0:1,...]
        saturation = pix_masks[:,1:2,...]
        hue = pix_masks[:,2:3,...]
        temp = pix_masks[:,3:4,...]
        lift = pix_masks[:,4:7,...]
        gamma = pix_masks[:,7:10,...]
        gain = pix_masks[:,10:13,...]
        shift = params[:,-3:,...]

        rendered_image_delta = self.render((input+1)/2., contrast, saturation, hue, temp, lift, gamma, gain, shift)

        return_dict = {"result":(rendered_image_delta + (input+1)/2. - 0.5) / 0.5}
        return return_dict

        #rendered_result = rendered_image_delta + input

    def forward_cluster(self, input, masks, params, cache=False):
        return self.graph_cluster(input, masks, params, cache)
        
    def mask_clustering(self, masks, params, k=5, consistent=False):
        return self.MaskCluster.mask_clustering(masks, params, k, consistent)

@ARCH_REGISTRY.register()
class Renderer_N10(nn.Module):
    """
    Basic Graph for Recolor.
    """

    def __init__(self, kernel_size=51, min_sigma=3, eps=1e-6):
        super(Renderer_N10, self).__init__()
        x_range = torch.linspace(-kernel_size//2,kernel_size-kernel_size//2, kernel_size)
        y_range = torch.linspace(-kernel_size//2,kernel_size-kernel_size//2, kernel_size)
        x, y = torch.meshgrid(x_range, y_range)
        self.x = nn.Parameter(x, requires_grad=False)
        self.y = nn.Parameter(y, requires_grad=False)
        self.eps = eps
        self.min_sigma = min_sigma
        self.kernel_size = kernel_size

        self.pad = nn.ReflectionPad2d((kernel_size-1)//2)

    def render(self, image, mask,
               contrast, saturation, hue, temp,
               lift, gamma, gain, shift):
        
        # contrast [0,...)
        contrast = contrast + 1
        delta = contrast * image + (1 - contrast) * torch.mean(image) - image

        # saturation [0,1]
        saturation = (saturation + 1) / 2.
        image_L = rgb2lab_func(image)[:,0:1,...]/100.0
        delta += 2 * saturation * image + (1 - 2 * saturation) * image_L - image

        # hue [0,1] 
        hue = (hue + 1) / 2.
        delta[:,0:1,...] += image[:,0:1,...] * (hue - 0.5) / 0.5 * 0.1
        delta[:,2:3,...] += image[:,2:3,...] * (hue - 0.5) / 0.5 * 0.1
        delta[:,1:2,...] += image[:,1:2,...] * (0.5 - hue) / 0.5 * 0.05

        # temp [0,1]
        temp = (temp + 1) / 2.
        delta[:,0:1,...] += image[:,0:1,...] * F.leaky_relu( 
            (temp - 0.5) / 0.5 * (253 - 128) / 128,
            (128 - 104) / (253 - 128)
        )
        delta[:,1:2,...] += image[:,1:2,...] * F.leaky_relu( 
            (temp - 0.5) / 0.5 * (99 - 128) / 128,
            0.0
        )
        delta[:,2:3,...] += image[:,2:3,...] * F.leaky_relu( 
            (temp - 0.5) / 0.5 * (16 - 128) / 128,
            (128 - 180) / (16 - 128)
        )

        # lift
        delta += 2*lift*(1 - image)

        # gain
        delta += 2*gain*image

        # gamma
        delta += gamma*7.2*(0.25-(image-0.5)**2)

        # shift
        delta[:,0:1,...] += shift[...,0:1]
        delta[:,1:2,...] += shift[...,1:2]
        delta[:,2:3,...] += shift[...,2:3]

        return delta * mask 

    def gaussian_blur_use_conv2d(self, mask, alpha):
        y = self.y.expand([mask.shape[0], 1, -1, -1])
        x = self.x.expand([mask.shape[0], 1, -1, -1])
        sigma = torch.clamp(alpha,0,1)*(self.kernel_size-self.min_sigma)+self.min_sigma
        kernel = 1./(2*3.1415926*sigma**2) * torch.exp(-(x**2 + y**2)/(2*3.1415926*sigma**2))
        kernel /= torch.sum(kernel,dim=(1,2,3),keepdims=True)
        output = torch.zeros_like(mask).to(mask.device)

        return F.conv2d(self.pad(mask.permute(1,0,2,3)),weight=kernel,groups=kernel.size(0)).permute(1,0,2,3)

    def forward(self, input, masks, params, cache=False, *args, **kwargs):
        assert masks.size(1) == params.size(1), f"n_masks=={masks.size(1)}, but params has {params.size(1)} channels."
        concat_graph = []
        for m in range(masks.size(1)):
            contrast = params[:,m:m+1,:,0:1]
            saturation = params[:,m:m+1,:,1:2]
            hue = params[:,m:m+1,:,2:3]
            temp = params[:,m:m+1,:,3:4]
            lift = params[:,m:m+1,:,4:5]
            gamma = params[:,m:m+1,:,5:6]
            gain = params[:,m:m+1,:,6:7]
            shift = params[:,m:m+1,:,7:10]
            alpha = params[:,m:m+1,:,10:11]

            mask = self.gaussian_blur_use_conv2d(masks[:,m:m+1,...], alpha) 
            #mask = masks[:,m:m+1,...]

            rendered_image_delta = self.render((input+1)/2., mask, contrast, saturation, hue, temp, lift, gamma, gain, shift)
            concat_graph.append(rendered_image_delta)

        return_dict = {
            "result": (sum(concat_graph) + (input+1)/2. - 0.5)/0.5
        }
        if cache:
            rendered_image = torch.cat([((delta + (input+1)/2.)-0.5)/0.5 for delta in concat_graph],dim=1)
            return_dict.update({'cache_imgs': rendered_image})
        return return_dict

@ARCH_REGISTRY.register()
class Renderer_N10_cascaded(nn.Module):
    """
    Basic Graph for Recolor.
    """

    def __init__(self, kernel_size=51, min_sigma=3, eps=1e-6):
        super(Renderer_N10_cascaded, self).__init__()
        x_range = torch.linspace(-kernel_size//2,kernel_size-kernel_size//2, kernel_size)
        y_range = torch.linspace(-kernel_size//2,kernel_size-kernel_size//2, kernel_size)
        x, y = torch.meshgrid(x_range, y_range)
        self.x = nn.Parameter(x, requires_grad=False)
        self.y = nn.Parameter(y, requires_grad=False)
        self.eps = eps
        self.min_sigma = min_sigma
        self.kernel_size = kernel_size

        self.pad = nn.ReflectionPad2d((kernel_size-1)//2)

    def render(self, image, mask, ordering,
               contrast, saturation, hue, temp,
               lift, gamma, gain, shift):

        for filter_name in ordering:
            # contrast [0,...)
            if filter_name == "contrast":
                contrast = contrast + 1
                image = contrast * image + (1 - contrast) * torch.mean(image)

            # saturation [0,1]
            elif filter_name == "saturation":
                saturation = (saturation + 1) / 2.
                image = image.clamp(0, 1)
                image_L = rgb2lab_func(image)[:,0:1,...]/100.0
                image = 2 * saturation * image + (1 - 2 * saturation) * image_L

            # hue [0,1] 
            elif filter_name == "hue":
                hue = (hue + 1) / 2.
                delta_0 = image[:,0:1,...] * (hue - 0.5) / 0.5 * 0.1
                delta_2 = image[:,2:3,...] * (hue - 0.5) / 0.5 * 0.1
                delta_1 = image[:,1:2,...] * (0.5 - hue) / 0.5 * 0.05
                image = image + torch.cat([delta_0,delta_1,delta_2], axis=1)

            # temp [0,1]
            elif filter_name == "temp":
                temp = (temp + 1) / 2.
                delta_0 = image[:,0:1,...] * F.leaky_relu( 
                    (temp - 0.5) / 0.5 * (253 - 128) / 128,
                    (128 - 104) / (253 - 128)
                )
                delta_1 = image[:,1:2,...] * F.leaky_relu( 
                    (temp - 0.5) / 0.5 * (99 - 128) / 128,
                    0.0
                )
                delta_2 = image[:,2:3,...] * F.leaky_relu( 
                    (temp - 0.5) / 0.5 * (16 - 128) / 128,
                    (128 - 180) / (16 - 128)
                )
                image = image + torch.cat([delta_0,delta_1,delta_2], axis=1)

            # lift
            elif filter_name == "lift":
                image = image + 2*lift*(1 - image)

            # gain
            elif filter_name == "gain":
                image = image + 2*gain*image

            # gamma
            elif filter_name == "gamma":
                image = image + gamma*7.2*(0.25-(image-0.5)**2)

            # shift
            elif filter_name == "shift":
                delta_0 = image[:,0:1,...] + shift[...,0:1]
                delta_1 = image[:,1:2,...] + shift[...,1:2]
                delta_2 = image[:,2:3,...] + shift[...,2:3]
                image = torch.cat([delta_0,delta_1,delta_2], axis=1)

            else:
                raise NotImplementedError(f"{filter_name} is not implemented.")


        return image * mask 

    def gaussian_blur_use_conv2d(self, mask, alpha):
        y = self.y.expand([mask.shape[0], 1, -1, -1])
        x = self.x.expand([mask.shape[0], 1, -1, -1])
        sigma = torch.clamp(alpha,0,1)*(self.kernel_size-self.min_sigma)+self.min_sigma
        kernel = 1./(2*3.1415926*sigma**2) * torch.exp(-(x**2 + y**2)/(2*3.1415926*sigma**2))
        kernel /= torch.sum(kernel,dim=(1,2,3),keepdims=True)
        output = torch.zeros_like(mask).to(mask.device)

        return F.conv2d(self.pad(mask.permute(1,0,2,3)),weight=kernel,groups=kernel.size(0)).permute(1,0,2,3)

    def forward(self, input, masks, params, cache=False, order=None):
        assert masks.size(1) == params.size(1), f"n_masks=={masks.size(1)}, but params has {params.size(1)} channels."
        concat_graph = []
        for m in range(masks.size(1)):
            contrast = params[:,m:m+1,:,0:1]
            saturation = params[:,m:m+1,:,1:2]
            hue = params[:,m:m+1,:,2:3]
            temp = params[:,m:m+1,:,3:4]
            lift = params[:,m:m+1,:,4:5]
            gamma = params[:,m:m+1,:,5:6]
            gain = params[:,m:m+1,:,6:7]
            shift = params[:,m:m+1,:,7:10]
            alpha = params[:,m:m+1,:,10:11]

            mask = self.gaussian_blur_use_conv2d(masks[:,m:m+1,...], alpha) 
            #mask = masks[:,m:m+1,...]

            if order is None:
                #order = ["lift","saturation","gamma","temp","gain","contrast","hue","shift"]
                raise IOError("Ordering of filter can't be none.")
            rendered_image = self.render(\
                (input+1)/2., mask, \
                order,
                contrast, saturation, hue, temp, lift, gamma, gain, shift)
            concat_graph.append(rendered_image)

        return_dict = {
            "result": (concat_graph[-1] - 0.5)/0.5
        }
        #if cache:
        #    rendered_image = torch.cat([((delta + (input+1)/2.)-0.5)/0.5 for delta in concat_graph],dim=1)
        #    return_dict.update({'cache_imgs': rendered_image})
        return return_dict

class Renderer_N1_K10(nn.Module):
    """
    Basic Graph for Recolor.
    """

    def __init__(self, eps=1e-15):
        super(Renderer_N1_K10, self).__init__()
        self.MaskCluster = MaskCluster(eps=eps)
        self.graph_cluster = Renderer_N10()

    def render(self, image, 
               contrast, saturation, hue, temp,
               lift, gamma, gain, shift):

        delta_list = []
        
        # contrast [0,...)
        contrast = contrast + 1
        delta = contrast * image + (1 - contrast) * torch.mean(image) - image
        delta_list.append( delta )

        # saturation [0,1]
        saturation = (saturation + 1) / 2.
        image_L = rgb2lab_func(image)[:,0:1,...]/100.0
        delta = 2 * saturation * image + (1 - 2 * saturation) * image_L - image
        delta_list.append( delta )

        # hue [0,1] 
        delta = torch.zeros_like( image )
        hue = (hue + 1) / 2.
        delta[:,0:1,...] += image[:,0:1,...] * (hue - 0.5) / 0.5 * 0.1
        delta[:,2:3,...] += image[:,2:3,...] * (hue - 0.5) / 0.5 * 0.1
        delta[:,1:2,...] += image[:,1:2,...] * (0.5 - hue) / 0.5 * 0.05
        delta_list.append( delta )

        # temp [0,1]
        delta = torch.zeros_like( image )
        temp = (temp + 1) / 2.
        delta[:,0:1,...] += image[:,0:1,...] * F.leaky_relu( 
            (temp - 0.5) / 0.5 * (253 - 128) / 128,
            (128 - 104) / (253 - 128)
        )
        delta[:,1:2,...] += image[:,1:2,...] * F.leaky_relu( 
            (temp - 0.5) / 0.5 * (99 - 128) / 128,
            0.0
        )
        delta[:,2:3,...] += image[:,2:3,...] * F.leaky_relu( 
            (temp - 0.5) / 0.5 * (16 - 128) / 128,
            (128 - 180) / (16 - 128)
        )
        delta_list.append( delta )

        # lift
        delta = 2*lift*(1 - image)
        delta_list.append( delta )

        # gain
        delta = 2*gain*image
        delta_list.append( delta )

        # gamma
        delta = gamma*7.2*(0.25-(image-0.5)**2)
        delta_list.append( delta )

        # shift
        delta = shift
        delta_list.append( delta )

        return delta_list 

    def forward(self, input, masks, params, cache=False, delta_pix_masks=None, *args, **kwargs):
        assert masks.size(1) == params.size(1) - 3, f"n_masks=={masks.size(1)}, but params has {params.size(1)} channels."
        pix_masks = masks * params[:,:-3,...]
        if delta_pix_masks is not None:
            pix_masks += delta_pix_masks
        contrast = pix_masks[:,0:1,...]
        saturation = pix_masks[:,1:2,...]
        hue = pix_masks[:,2:3,...]
        temp = pix_masks[:,3:4,...]
        lift = pix_masks[:,4:5,...]
        gamma = pix_masks[:,5:6,...]
        gain = pix_masks[:,6:7,...]
        shift = params[:,7:10,...]

        rendered_image_delta_list = self.render((input+1)/2., contrast, saturation, hue, temp, lift, gamma, gain, shift)

        return_dict = {"result":(sum(rendered_image_delta_list) + (input+1)/2. - 0.5) / 0.5}
        if cache:
            rendered_image = torch.cat([((delta + (input+1)/2.)-0.5)/0.5 for delta in rendered_image_delta_list],dim=1)
            rendered_image = torch.cat([rendered_image,(sum(rendered_image_delta_list[:-1]) + (input+1)/2. - 0.5) / 0.5],dim=1)
            return_dict.update({'cache_imgs': rendered_image})
        return return_dict

        #rendered_result = rendered_image_delta + input

    def forward_cluster(self, input, masks, params, cache=False):
        return self.graph_cluster(input, masks, params, cache)
        
    def mask_clustering(self, masks, params, k=5, consistent=False):
        return self.MaskCluster.mask_clustering(masks, params, k, consistent)

class MaskCluster(object):
    def __init__(self, eps=1e-15):
        self.eps = eps

    def mask_clustering(self, masks, params, k=5, consistent=False):
        assert masks.size(0) == 1, f"Mask clustering requires mask.size(0) == 1, but find {mask.size(0)}"

        pix_masks = masks * params[:,:-3,...]
        #global_p = torch.mean(pix_masks, dim=(2,3), keepdim=True)
        #pix_masks -= global_p
        params = torch.cat([params, torch.Tensor([[0.0]]).unsqueeze(2).unsqueeze(3).to(masks.device)],dim=1)

        pix_norm = torch.max(torch.abs(pix_masks),dim=1,keepdim=True)[0] + self.eps
        pix_masks = pix_masks / pix_norm
        #return {"params": pix_masks}
        pix_masks = pix_masks.reshape(pix_masks.size(1),-1).permute(1,0)

        # clustering
        if consistent:
            if hasattr(self, 'cluster_centers'):
                cluster_ids_mask = kmeans_predict(
                    pix_masks, self.cluster_centers, 'cosine', pix_masks.device)
                cluster_centers = self.cluster_centers
            else:
                cluster_ids_mask, cluster_centers = kmeans(
                    X=pix_masks, num_clusters=k, distance='cosine', device=pix_masks.device)
                self.cluster_centers = cluster_centers
        else:
            cluster_ids_mask, cluster_centers = kmeans(
                X=pix_masks, num_clusters=k, distance='cosine', device=pix_masks.device)

        group_mask = cluster_ids_mask.reshape(1,masks.size(2),masks.size(3)).to(pix_masks.device)
        mask = torch.cat([group_mask==i for i in range(k)],dim=0).type(torch.float32).unsqueeze(0)

        local_p = cluster_centers.unsqueeze(1).unsqueeze(0).to(pix_masks.device)

        for i in range(mask.size(1)):
            region_mask = mask[:,i:i+1,...]
            norm_p = torch.max(pix_norm * region_mask)
            region_mask = pix_norm / norm_p * region_mask
            mask[:,i:i+1,...] = region_mask

            local_p[0,i,0,:] *= norm_p
        #print(local_p)

        return_dict = {
            "masks": mask,
            "params": torch.cat([local_p, torch.cat([params[:,-4:,...].permute(0,2,3,1)]*k,dim=1)], dim=3)
            #"global_params": torch.cat([global_p,params[:,-3:,...]],dim=1)
        }

        return return_dict


