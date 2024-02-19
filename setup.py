import sys
import warnings
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

interactive = [
    'notebook==5.7.10',
    'jupyter_contrib_nbextensions',
    'jupyter_nbextensions_configurator',
    'matplotlib',
    'networkx',
    'seaborn',
    'plotly',
    'pandas',
    'prince',  # for Correspondence Analysis
    'nltk'  # for notebook of text processing
]
metrics = ['sparsebm==1.3']  # for Co-clustering ARI (CARI) only
tests = ['pytest', 'jinja2==3.1.3']

if (sys.version_info.major, sys.version_info.minor) != (3, 7):
    warnings.warn(
        'Python version is different from 3.7 -> spherecluster package '
        'will not be installed, so initializations with spherical k-means '
        'will not be possible.'
    )
    initialization = ['scikit-learn']
    base = [
        'numpy',
        'scipy',
        'numba'
    ]
else:
    warnings.warn(
        'Python version is 3.7 -> spherecluster package can be installed '
        'so initializations with spherical k-means are possible'
    )
    initialization = ['spherecluster', 'scikit-learn==0.20']
    base = [
        'numpy==1.21',
        'scipy',
        'numba'
    ]

all_extras = interactive + initialization + metrics + tests

setup(
    name='dcblockmodels',
    version='1.0.0',
    description='Dynamic and constrained block models',
    install_requires=base,
    extras_require={
        'interactive': interactive,
        'intialization': initialization,
        'metrics': metrics,
        'all': all_extras
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    tests_require=tests,
    packages=find_packages()
)
