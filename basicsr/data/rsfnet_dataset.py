import numpy as np
import random
import time
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import cv2
import os.path as osp

from basicsr.data.data_util import paired_paths_from_folder, \
    paired_paths_from_lmdb, \
    paired_paths_from_meta_info_file, \
    multi_paired_paths_from_folder, \
    multi_paired_paths_from_folder_without_global_gt, \
    multi_unpaired_paths_from_folder_without_global_gt, \
    multi_paired_paths_from_meta_info_file_without_global_gt, \
    manytoone_paths_from_meta_info_file_without_global_gt, \
    multi_mask_multi_paired_paths_from_meta_info_file_without_global_gt
from basicsr.data.transforms import augment, \
    paired_random_crop, multi_paired_random_crop, \
    multi_paired_random_crop_without_global, \
    multi_unpaired_random_crop_without_global
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class RSFNetDataset(data.Dataset):
    """Multi Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(RSFNetDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.mask_num = opt['mask_num'] if 'mask_num' in opt else None
        self.mask_as_label = opt.get('mask_as_label', False)

        self.gt_folder, self.lq_folder, self.mask_folder = \
            opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_mask']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        if 'filename_tmpl_mask' in opt:
            self.filename_tmpl_mask = opt['filename_tmpl_mask']
        else:
            self.filename_tmpl_mask = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = multi_paired_paths_from_meta_info_file_without_global_gt(\
                [self.lq_folder, self.gt_folder, self.mask_folder], \
                ['lq', 'gt', 'mask'], \
                self.opt['meta_info_file'], self.filename_tmpl, self.filename_tmpl_mask,\
                self.mask_num)
        else:
            self.paths = multi_paired_paths_from_folder_without_global_gt(\
                [self.lq_folder, self.gt_folder, self.mask_folder], \
                ['lq', 'gt', 'mask'], \
                self.filename_tmpl, self.filename_tmpl_mask, \
                self.mask_num)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        mask_paths = self.paths[index]['mask_path']
        img_masks = []
        for mask_path in mask_paths:
            img_bytes = self.file_client.get(mask_path, 'mask')
            img_masks.append( imfrombytes(img_bytes, float32=True)[:,:,0:1] )
            if img_masks[-1].shape[:-1] != img_gt.shape[:-1]:
                img_masks[-1] = np.expand_dims(cv2.resize(img_masks[-1], (img_gt.shape[1],img_gt.shape[0]), interpolation=cv2.INTER_NEAREST), axis=2)

        # generate 0-1 masks or labels
        if self.mask_as_label:
            img_masks_all = np.zeros_like(img_lq[:,:,0:1]) + self.mask_num
            for i in range(len(mask_paths)):
                mask_id = osp.splitext(osp.basename(mask_paths[i]))[0].split("_")[-1]
                img_masks_all[img_masks[i]==1.0] = int(mask_id)
        else:
            img_masks_all = np.zeros((img_lq.shape[0],img_lq.shape[1],self.mask_num))
            for i in range(len(mask_paths)):
                mask_id = int(osp.splitext(osp.basename(mask_paths[i]))[0].split("_")[-1])
                img_masks_all[:,:,mask_id:mask_id+1] = img_masks[i]
        img_masks = img_masks_all

        # augmentation for training
        if self.opt['phase'] == 'train':
            # random resize before crop
            resize_before_crop = self.opt.get('resize_before_crop')
            if resize_before_crop is not None:
                if not isinstance(resize_before_crop, list):
                    resize_before_crop = [resize_before_crop,resize_before_crop]
            random_resize_before_crop = self.opt.get('random_resize_before_crop')
            if random_resize_before_crop is not None:
                if not isinstance(random_resize_before_crop, list):
                    random_resize_before_crop = [random_resize_before_crop,random_resize_before_crop]
                s_ratio = random_resize_before_crop[0]+random.random()*(random_resize_before_crop[1]-random_resize_before_crop[0])
                s_ratio = max(256./min(img_lq.shape[:-1]), s_ratio)
                resize_before_crop = [int(img_lq.shape[0]*s_ratio),int(img_lq.shape[1]*s_ratio)]
            if resize_before_crop is not None:
                img_gt = cv2.resize(img_gt, resize_before_crop[::-1], interpolation=cv2.INTER_LINEAR) 
                img_lq = cv2.resize(img_lq, resize_before_crop[::-1], interpolation=cv2.INTER_LINEAR) 
                if len(img_masks.shape) < 3:
                    img_masks = np.expand_dims(cv2.resize(img_masks, resize_before_crop[::-1], interpolation=cv2.INTER_LINEAR), axis=2) 
            # random crop
            gt_size = self.opt['gt_size']
            img_gt, img_lq, img_masks = multi_paired_random_crop_without_global(img_gt, img_lq, img_masks, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq, img_masks = augment([img_gt, img_lq, img_masks], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_masks = img2tensor([img_gt, img_lq, img_masks], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            if not self.mask_as_label:
                normalize(img_masks, 0., 1., inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'mask': img_masks, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

