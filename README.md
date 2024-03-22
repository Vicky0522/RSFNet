## RSFNet

Official PyTorch Implementation of ICCV 2023 Paper "[RSFNet: A White-Box Image Retouching Approach using Region-Specific Color Filters](https://arxiv.org/abs/2303.08682)"

[![arXiv](https://img.shields.io/badge/arXiv-2303.08682-b31b1b.svg)](https://arxiv.org/abs/2303.08682)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=Vicky0522/RSFNet)

https://github.com/Vicky0522/RSFNet/assets/18029603/7ddb5edb-bfe4-416a-935e-d7e95ad650ec

https://github.com/Vicky0522/RSFNet/assets/18029603/6fa61244-cd48-4562-9cb8-20eef0422d9e

https://github.com/Vicky0522/RSFNet/assets/18029603/821fbf3d-0053-4de3-98a2-f33ea01138b8

https://github.com/Vicky0522/RSFNet/assets/18029603/92a299a0-6e55-4748-ac02-df4920faadfe

## Requirements

- Python = 3.6
- CUDA = 10.2
- Pytorch = 1.7.1

Build the environment by running
```
conda create -n RSFNet python=3.6
conda activate RSFNet

cd RSFNet
python setup.py develop
```

## Inference

### 1. Download the pretrained model.

| Model                 | Description                          |  Note |
| --------------------- | :----------------------------------- | :-----|
| [RSFNet_map_fivek_zwc.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_map_fivek_zwc.pth) | RSFNet-map trained on MIT-Adobe5k, using `Input Zeroed with expert C` as input | Trained with the main framework |
| [RSFNet_map_fivek_zas.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_map_fivek_zas.pth) | RSFNet-map trained on MIT-Adobe5k, using `Input Zeroed as Shot` as input | Trained with the main framework  |
| [RSFNet_map_ppr10k_a.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_map_ppr10k_a.pth)   | RSFNet-map trained on PPR10K, using`target_a` as input | Trained with the main framework |
| [RSFNet_map_ppr10k_b.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_map_ppr10k_b.pth)   | RSFNet-map trained on PPR10K, using`target_b` as input | Trained with the main framework |
| [RSFNet_map_ppr10k_c.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_map_ppr10k_c.pth)   | RSFNet-map trained on PPR10K, using`target_c` as input | Trained with the main framework |
| [RSFNet_saliency_fivek_zwc.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_saliency_fivek_zwc.pth)   | RSFNet-saliency trained on MIT-Adobe5k, using`Input Zeroed with expert C` as input | Trained with saliency masks |
| [RSFNet_saliency_fivek_zas.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_saliency_fivek_zas.pth)   | RSFNet-saliency trained on MIT-Adobe5k, using`Input Zeroed as Shot` as input | Trained with saliency masks |
| [RSFNet_saliency_ppr10k_a.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_saliency_ppr10k_a.pth)   | RSFNet-saliency trained on PPR10K, using`target_a` as input | Trained with saliency masks |
| [RSFNet_saliency_ppr10k_b.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_saliency_ppr10k_b.pth)   | RSFNet-saliency trained on PPR10K, using`target_b` as input | Trained with saliency masks |
| [RSFNet_saliency_ppr10k_c.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_saliency_ppr10k_c.pth)   | RSFNet-saliency trained on PPR10K, using`target_c` as input | Trained with saliency masks |
| [RSFNet_palette_fivek_zwc.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_palette_fivek_zwc.pth)   | RSFNet-palette trained on MIT-Adobe5k, using`Input Zeroed with expert C` as input | Trained with palette-based masks |
| [RSFNet_palette_fivek_zas.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_palette_fivek_zas.pth)   | RSFNet-palette trained on MIT-Adobe5k, using`Input Zeroed as Shot` as input | Trained with palette-based masks |
| [RSFNet_global_fivek_zwc.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_global_fivek_zwc.pth)   | RSFNet-global trained on MIT-Adobe5k, using`Input Zeroed with expert C` as input | Trained without masks, only global transformation is applied |
| [RSFNet_global_fivek_zas.pth](https://huggingface.co/Vicky0522/RSFNet-models/blob/main/RSFNet_global_fivek_zas.pth)   | RSFNet-global trained on MIT-Adobe5k, using`Input Zeroed as Shot` as input | Trained without masks, only global transformation is applied |


Then put them under the created folder `pretrained`

### 2. Run

Choose the config file under the folder `options/infer/RSFNet`, change `dataroot_lq` and `pretrain_network_g` to your local data path and checkpoint path, then run:
```
python basicsr/test.py -opt options/RSFNet/infer_xxx.yml
```

## Train

### 1. Prepare the dataset.

Download [MIT-Adobe5k](https://data.csail.mit.edu/graphics/fivek/) and [PPR10K](https://github.com/csjliang/PPR10K) from the official website. 

Note that `Adobe-Lighroom` is needed for pre-processing the MIT-Adobe5k dataset. For convenience, we provide processed data and corresponding masks on [our Hugging Face page](https://huggingface.co/datasets/Vicky0522/MIT-Adobe5k-for-RSFNet).

You can also generate masks from the scratch using the following steps:

#### Generate saliency masks

Download pre-trained model [PoolNet-ResNet50 w/o edge model](https://drive.google.com/open?id=12Zgth_CP_kZPdXwnBJOu4gcTyVgV2Nof) from [PoolNet](https://github.com/backseason/PoolNet). Then run:
```
cd datasets/generate
python saliency_mask_generate.py --input_dir /path_to_your_input/ --output_dir /path_to_your_output/ --gpu_id 0 --weight /path_to_weight/
```

#### Generate palette-based masks

We use [Palette-based Photo Recoloring](https://github.com/pipi3838/DIP_Final) to generate palettes. To generate palette-based masks, run:

```
cd datasets/generate
python palette_mask_generate.py --input_dir /path_to_your_input/ --output_dir /path_to_your_output/ 
```

Put all the datasets under the created folder `datasets`. The folder structure should look like this:
```
- datasets
  - Adobe5k
    - train.txt
    - test.txt
    - expertC
      - a0000.png
      ...
    - zeroed_with_expertC 
      - a0000.png
      ...
    - zeroed_as_shot
      - a0000.png
      ...
    - saliency
      - a0000_mask_0.png
      ...
    - palette
      - a0000_mask_0.png
      ...
      - a0000_mask_4.png
      ...
    - semantic
      - a0000_mask_0.png
      ...
      - a0000_mask_4.png
      ...
  - PPR10K
    - train.txt
    - test.txt
    - mask_360p
      - 0_0_0.png
      ...
    - train_val_images_tif_360p/
```

### 2. Visualize the training process.

First, run the visdom server:
```
python -m visdom.server -port xxxx
```
Then uncomment and modify the block in config file:
```
# visdom settings
#visuals:
#  show_freq: 50
#  display_port: xxxx
#  display_server: http://localhost
#  path: experiments/rsfnet_xxx/visuals
```
Visit http://localhost:xxxx via your internet browser and watch the training process.

You can leave the visdom block commented if you wish not to visualize the training process.

### 3. Run
```
python basicsr/train.py -opt options/train/RSFNet/train_xxx.yml
```

## Evaluation

```
python basicsr/test.py -opt options/test/RSFNet/test_xxx.yml
```

## Citation

If you find our work is helpful for your research, please consider citing:

```
@article{oywq2023rsfnet,
  title={RSFNet: A white-Box image retouching approach using region-specific color filters},
  author={Wenqi Ouyang and Yi Dong and Xiaoyang Kang and Peiran Ren and Xin Xu and Xuansong Xie},
  journal={https://arxiv.org/abs/2303.08682},
  year={2023}
}
```

## License
© Alibaba, 2023. For academic and non-commercial use only.

## Acknowledgments
We thank the authors of BasicSR for the awesome training pipeline.

> Xintao Wang, Ke Yu, Kelvin C.K. Chan, Chao Dong and Chen Change Loy. BasicSR: Open Source Image and Video Restoration Toolbox. https://github.com/xinntao/BasicSR, 2020.

Code for the visualization is partially borrowed from [CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 

Code for the network architecture is partially borrowed from [mmdetection](https://github.com/open-mmlab/mmdetection)



