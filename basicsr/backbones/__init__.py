import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import BACKBONE_REGISTRY

__all__ = ['build_backbone']

# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
backbone_folder = osp.dirname(osp.abspath(__file__))
backbone_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(backbone_folder) if v.endswith('_backbone.py')]
# import all the arch modules
_backbone_modules = [importlib.import_module(f'basicsr.backbones.{file_name}') for file_name in backbone_filenames]


def build_backbone(opt):
    opt = deepcopy(opt)
    backbone_type = opt.pop('type')
    net = BACKBONE_REGISTRY.get(backbone_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Backbone [{net.__class__.__name__}] is created.')
    return net
