from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='CB1-ntwk-modeling',
    version='1.0',
    description='CB1-ntwk-modeling',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/yzerlaut/CB1_ntwk_modeling',
    author='Yann Zerlaut',
    author_email='yann.zerlaut@icm-institute.org',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ],
    keywords='vision physiology',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "brian2",
        "scipy",
        "elephant"
    ]
)
