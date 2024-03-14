from setuptools import setup
from setuptools import find_packages

exclude_dirs = ("configs",)

# for install, do: pip install -ve .

# Install Base Environment:

setup(
    name='xmuda',
    version="0.0.1",
    url="https://github.com/maxjaritz/xmuda_private",
    description="xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation",
    install_requires=['yacs', 'nuscenes-devkit', 'tabulate', 'tensorboard'],
    packages=find_packages(exclude=exclude_dirs),
)