import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import time
import numpy as np

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class RSFNetMaskInModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(RSFNetMaskInModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('mask_opt'):
            self.cri_mask = build_loss(train_opt['mask_opt']).to(self.device)
        else:
            self.cri_mask = None

        if train_opt.get('cache_opt'):
            self.cri_cache = build_loss(train_opt['cache_opt']).to(self.device)
        else:
            self.cri_cache = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_mask is None and self.cri_cache is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'gt2' in data:
            self.gt2 = data['gt2'].to(self.device)
        if 'mask' in data:
            self.mask_gt = data['mask'].to(self.device)
        if 'in_params' in data:
            self.in_params = data['in_params'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        kwargs = {}
        if hasattr(self, 'mask_gt'):
            kwargs.update({'masks': self.mask_gt})
        results = self.net_g(self.lq, **kwargs)
        if results.get('result') is not None:
            self.output = results.get('result')
        if results.get('cache') is not None:
            self.cache = results.get('cache')

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # mask loss
        if self.cri_mask:
            try:
                l_mask = self.cri_mask(self.mask[:,:self.mask_gt.size(1),...], self.mask_gt)
            except:
                l_mask = self.cri_mask(self.mask)
            l_total += l_mask
            loss_dict['l_mask'] = l_mask
        # cache loss
        if self.cri_cache:
            l_cache = self.cri_cache(self.cache, self.gt2, self.gt, self.mask_gt)
            l_total += l_cache
            loss_dict['l_cache'] = l_cache

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        kwargs = {}
        if hasattr(self, 'in_params'):
            kwargs.update({'params': self.in_params})
        if hasattr(self, 'mask_gt'):
            kwargs.update({'masks': self.mask_gt})
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                begin = time.time()
                results = self.net_g_ema(self.lq, **kwargs)
                self.time_cost = time.time() - begin
                if results.get('result') is not None:
                    self.output = results.get('result')
                if results.get('cache') is not None:
                    self.cache = results.get('cache')
                if results.get('params') is not None:
                    self.params = results.get('params')
        else:
            self.net_g.eval()
            with torch.no_grad():
                begin = time.time()
                results = self.net_g(self.lq, **kwargs)
                self.time_cost = time.time() - begin
                if results.get('result') is not None:
                    self.output = results.get('result')
                if results.get('cache') is not None:
                    self.cache = results.get('cache')
                if results.get('params') is not None:
                    self.params = results.get('params')
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        save_params = self.opt['val'].get('save_params', False)
        save_mask = self.opt['val'].get('save_mask', False)
        save_cache = self.opt['val'].get('save_cache', False)
        save_metrics_per_image = self.opt['val'].get('save_metrics_per_image', False)
        time_cost = self.opt['val'].get('time_cost', False)
        if time_cost:
            total_time_cost = []
            total_time_cost_split = []
        params_tensor_all = None

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}
        if save_metrics_per_image:
            self.metric_results_per_image = {metric: [] for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            if time_cost:
                total_time_cost.append(self.time_cost)
                if hasattr(self.net_g, "time_cost"):
                    total_time_cost_split.append(self.get_g.time_cost)

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], min_max=(-1,1))
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], min_max=(-1,1))
                metric_data['img2'] = gt_img
                del self.gt
            if 'mask_gt' in visuals:
                mask_groups = torch.split(visuals['mask_gt'], 1, dim=1)
                mask_img = [tensor2img([mask], min_max=(0,1)) for mask in mask_groups]
                del self.mask_gt
                del mask_groups
            if 'cache' in visuals:
                cache_groups = torch.split(visuals['cache'], 3, dim=1)
                cache_img = [tensor2img([cache], min_max=(-1,1)) for cache in cache_groups]
                del self.cache
                del cache_groups

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if save_mask:

                if 'mask_gt' in visuals:

                    for i in range(len(mask_img)):
                        mask = mask_img[i]
                        if self.opt['is_train']:
                            save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                     f'{img_name}_mask{i}_{current_iter}.png')
                        else:
                            if self.opt['val']['suffix']:
                                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                         f'{img_name}_mask{i}_{self.opt["val"]["suffix"]}.png')
                            else:
                                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                         f'{img_name}_mask{i}_{self.opt["name"]}.png')
                        imwrite(mask, save_img_path)

            if save_cache:

                if 'cache' in visuals:

                    for i in range(len(cache_img)):
                        cache = cache_img[i]
                        if self.opt['is_train']:
                            save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                     f'{img_name}_cache{i}_{current_iter}.png')
                        else:
                            if self.opt['val']['suffix']:
                                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                         f'{img_name}_cache{i}_{self.opt["val"]["suffix"]}.png')
                            else:
                                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                         f'{img_name}_cache{i}_{self.opt["name"]}.png')
                        imwrite(cache, save_img_path)

            if save_params and 'params' in visuals:
                if params_tensor_all is None:
                    params_tensor_all = torch.zeros((len(dataloader),visuals['params'].size(1),1,visuals['params'].size(3)))
                params_tensor_all[idx:idx+1,...] = visuals['params']
                    
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += tmp
                    if save_metrics_per_image:
                        self.metric_results_per_image[name].append( [val_data.get('mask_path', [""]), tmp] ) 

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if time_cost:
            print("Average time cost for [%d] images: [%.15f ms]" % (\
                len(dataloader), 1000*sum(total_time_cost) / len(total_time_cost)))
            logger = get_root_logger()
            logger.info("Average time cost for [%d] images: [%.15f ms]" % (\
                len(dataloader), 1000*sum(total_time_cost) / len(total_time_cost)))
            print(f"Average time cost(split) for {len(dataloader)} images: {np.mean(1000*np.array(total_time_cost_split), axis=0)}")

        if save_params and 'params' in visuals:
            if self.opt['is_train']:
                torch.save(params_tensor_all.cpu(), osp.join(self.opt['path']['visualization'],
                               f'params_{dataset_name}_{current_iter}.pth'))
            else:
                if self.opt['val']['suffix']:
                    torch.save(params_tensor_all.cpu(), osp.join(self.opt['path']['visualization'],
                               f'params_{dataset_name}_{self.opt["val"]["suffix"]}.pth'))
                else:
                    torch.save(params_tensor_all.cpu(), osp.join(self.opt['path']['visualization'],
                               f'params_{dataset_name}_{self.opt["name"]}.pth'))
            
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            if save_metrics_per_image:
                for metric in self.metric_results_per_image.keys():
                    if self.opt['is_train']:
                        fname = f'metric.{metric}_{dataset_name}_{current_iter}.txt'
                    else:
                        if self.opt['val']['suffix']:
                            fname = f'metric.{metric}_{dataset_name}_{self.opt["val"]["suffix"]}.txt'
                        else:
                            fname = f'metric.{metric}_{dataset_name}_{self.opt["name"]}.txt'
                    with open(osp.join(self.opt['path']['visualization'], fname), "w") as fw:
                        values = []
                        for item in self.metric_results_per_image[metric]:
                            fw.write(f'{item[0][0]} {item[1]}\n')
                            values.append( item[1] )
                        _mean = sum(values) / len(values)
                        _std = ( sum((np.array(values) - _mean)**2) / len(values) )**0.5
                        print("_std:", _std)
                    
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach()
        out_dict['result'] = self.output.detach()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach()
        if hasattr(self, 'mask_gt'):
            out_dict['mask_gt'] = self.mask_gt.detach()
        if hasattr(self, 'cache'):
            out_dict['cache'] = self.cache.detach()
        if hasattr(self, 'params'):
            out_dict['params'] = self.params.detach()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    @torch.no_grad()
    def inference(self, image, params=None, masks=None):
        assert isinstance(image, torch.Tensor), "image must be torch.Tensor"
        image = (image - 0.5)/0.5
        if params is None:
            input_data = {"lq": image, 'mask': masks}
        else:
            input_data = {"lq": image, 'in_params': params, 'mask': masks}

        self.feed_data(input_data)
        self.test()
        
        visuals = self.get_current_visuals()

        return_dict = {}
        if visuals.get('result') is not None:
            return_dict.update( {'result': (visuals['result'] + 1) / 2.} )
        if visuals.get('params') is not None:
            return_dict.update( {'params': visuals['params']} )
        if visuals.get('mask') is not None:
            return_dict.update( {'mask':  visuals['mask']} )
        return return_dict
