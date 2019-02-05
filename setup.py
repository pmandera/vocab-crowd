#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from setuptools import setup, find_packages, Extension

setup(
    name = "vocabtest",
    version = "0.1.0",
    packages = find_packages(),

    # install_requires = [# 'csv>=1.0', 'fs>=0.5.0',
    #                     'numpy>=1.9.1', 'scipy>=0.14.0',
    #                     'pandas>=0.15.1', 'scikit-learn>=0.15.0'],

    # metadata for upload to PyPI
    author = "Paweł Mandera",
    author_email = "pawel.mandera@ugent.be",
    description = "This is a package to work with results of vocabulary tests.",
    keywords = "vocab test",

    scripts = ['bin/vocabtest-subset'],
)
