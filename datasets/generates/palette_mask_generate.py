import os
import cv2
import numpy as np
import time
import tqdm

import torch
import torch.nn.functional as F

from basicsr.utils import (tensor2img, img2tensor)

from PaletteGenerate.palette_generate import PaletteGenerate


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu_id', type=int, default=0, help='cuda device id') 
    parser.add_argument('--input_dir', type=str, default='./input', help='path for input images')
    parser.add_argument('--output_dir', type=str, default='./output', help='path for output images')

    model = PaletteGenerate(gpu_id=args.gpu_id, k=5)

    image_path = args.input_dir
    mask_path = args.output_dir

    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    filelist = os.listdir(image_path)
    #filelist = sorted(filelist, key = lambda x: int(x[1:5]))
    pbar = tqdm(total=len(filelist), unit='image') 
    for filepath in filelist:
        # read images
        image = cv2.imread(os.path.join(image_path, filepath))
        image = img2tensor(image, True).to("cuda:{}".format(gpu_id))

        result, palette = model.forward(image)
        palette_img = tensor2img(palette, True, min_max=(0,1))

        cv2.imwrite(os.path.join(mask_path, os.path.splitext(filepath)[0]+"_palette.png"), palette_img)
        for i in range(result.size(1)):
            mask = pyutils.tensor2img(torch.cat([result[:,i:i+1,...],]*3,dim=1), False, min_max=(0,1))
            cv2.imwrite(os.path.join(mask_path, os.path.splitext(filepath)[0]+"_mask_{}.png".format(i)), mask)

        pbar.update(1)
        pbar.set_description(f'Generate palette-based mask for {filepath}')

    pbar.close()

        






        
    
        

