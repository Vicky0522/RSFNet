import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import HEAD_REGISTRY

__all__ = ['build_head']

# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
head_folder = osp.dirname(osp.abspath(__file__))
head_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(head_folder) if v.endswith('_head.py')]
# import all the arch modules
_head_modules = [importlib.import_module(f'basicsr.heads.{file_name}') for file_name in head_filenames]


def build_head(opt):
    opt = deepcopy(opt)
    head_type = opt.pop('type')
    net = HEAD_REGISTRY.get(head_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Head [{net.__class__.__name__}] is created.')
    return net
