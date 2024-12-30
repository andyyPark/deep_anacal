import os
from setuptools import setup, find_packages

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "deep_anacal",
    "_version.py",
)
with open(pth, 'r') as fp:
    exec(fp.read())

setup(
    name='deep_anacal',
    description="code for anacal w/ wide- and deep-field data",
    author="Andy Park",
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
)
