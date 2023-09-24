import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import NECK_REGISTRY

__all__ = ['build_neck']

# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
neck_folder = osp.dirname(osp.abspath(__file__))
neck_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(neck_folder) if v.endswith('_neck.py')]
# import all the arch modules
_neck_modules = [importlib.import_module(f'basicsr.necks.{file_name}') for file_name in neck_filenames]


def build_neck(opt):
    opt = deepcopy(opt)
    neck_type = opt.pop('type')
    net = NECK_REGISTRY.get(neck_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Neck [{net.__class__.__name__}] is created.')
    return net
