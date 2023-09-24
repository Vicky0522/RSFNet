import time
from setuptools import setup, find_packages
import sys
import os
import os.path as osp

WORK_DIR = "basicsr"
NAME = "basicsr"
author = "wenqi.oywq"
author_email = 'wenqi.oywq@alibaba-inc.com'

version_file = 'basicsr/version.py'

def get_hash():
    if False:#os.path.exists('.git'):
        sha = get_git_hash()[:7]
    # currently ignore this
    # elif os.path.exists(version_file):
    #     try:
    #         from basicsr.version import __version__
    #         sha = __version__.split('+')[-1]
    #     except ImportError:
    #         raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'

    return sha

def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
__gitsha__ = '{}'
version_info = ({})
"""
    sha = get_hash()
    with open('VERSION', 'r') as f:
        SHORT_VERSION = f.read().strip()
    VERSION_INFO = ', '.join([x if x.isdigit() else f'"{x}"' for x in SHORT_VERSION.split('.')])

    version_file_str = content.format(time.asctime(), SHORT_VERSION, sha, VERSION_INFO)
    with open(version_file, 'w') as f:
        f.write(version_file_str)

REQUIRE = [
    'tensorboard==2.10.1',
    'visdom==0.2.4',
    'lpips==0.1.4',
    'dists-pytorch==0.1',
    'kmeans_pytorch==0.3',
    'opencv-python==4.5.4.60',
    'scikit-image==0.16.2',
    'PyYAML==5.1',
    'dominate==2.7.0',
    'Ninja==1.11.1',
    'torch==1.7.1 torchvision==0.8.2 --index-url https://download.pytorch.org/whl/cu102'
]
    
def install_requires(REQUIRE):
    for item in REQUIRE:
        os.system(f'pip install {item}')

write_version_py()
install_requires(REQUIRE)
setup(
    name=NAME,
    packages=find_packages(),
    version=get_version(),
    description="training tools",
    author=author,
    author_email=author_email,
    keywords=["training tools"],
    install_requires=['tensorboard'],
    include_package_data=False,
    exclude_package_data={'':['.gitignore','README.md','./options/'],
                         },
    #package_data = {'VideoRecolor':\
    #                [
    #                 osp.join('data', '*.txt'), \
    #                 osp.join('data', 'filter', '*_lut.txt'), \
    #                 osp.join('data', 'HDR', '*_lut.txt'), \
    #                ]
    #               },
    entry_points={'console_scripts': ['pyinstrument = pyinstrument.__main__:main']},
    zip_safe=False,
)
