import os.path as osp

import torch
from torch.utils.cpp_extension import load

try:
    from cudaops_color import (rgb2lab_srgb_forward, rgb2lab_srgb_backward)
except ImportError:
    CUR_DIR = osp.abspath(osp.dirname(__file__))  
    cudaops_color = load( 
        name='cudaops_color',
        sources=[  
            osp.join(CUR_DIR, 'src/color_cuda.cpp'),
            osp.join(CUR_DIR, 'src/color_kernel.cu')
        ],
        verbose=True)
    from cudaops_color import (rgb2lab_srgb_forward, rgb2lab_srgb_backward)

class RGB2LABTransfer(torch.autograd.Function): 

    @staticmethod
    def forward(ctx, input):
        
        ctx.save_for_backward(input)

        return rgb2lab_srgb_forward(input.permute(0,2,3,1).contiguous()).permute(0,3,1,2)

    @staticmethod
    def backward(ctx, output_grad):
        input = ctx.saved_tensors[0]
        
        image_grad = rgb2lab_srgb_backward( \
            output_grad.permute(0,2,3,1).contiguous(), \
            input.permute(0,2,3,1).contiguous(), \
        )
        return image_grad.permute(0,3,1,2)

class LAB2RGBTransfer(torch.autograd.Function): 

    @staticmethod
    def forward(ctx, input):
        
        ctx.save_for_backward(input)

        return lab2rgb_srgb_forward(input.permute(0,2,3,1).contiguous()).permute(0,3,1,2)

    @staticmethod
    def backward(ctx, output_grad):
        input = ctx.saved_tensors[0]
        
        image_grad = lab2rgb_srgb_backward( \
            output_grad.permute(0,2,3,1).contiguous(), \
            input.permute(0,2,3,1).contiguous(), \
        )
        return image_grad.permute(0,3,1,2)

def rgb2lab_func(img: torch.Tensor) -> torch.Tensor:
    return RGB2LABTransfer.apply(img)

def lab2rgb_func(img: torch.Tensor) -> torch.Tensor:
    return LAB2RGBTransfer.apply(img)
