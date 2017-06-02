#!/usr/bin/env python
# -*- coding: utf-8 -*-

# {# pkglts, pysetup.kwds
# format setup arguments

from setuptools import setup, find_packages


short_descr = "An openalea package to evaluate the quality of segmented tissue images."
readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')


# find version number in src/vplants/segmentation_evaluation/version.py
version = {}
with open("src/vplants/segmentation_evaluation/version.py") as fp:
    exec(fp.read(), version)


setup_kwds = dict(
    name='vplants.segmentation_evaluation',
    version=version["__version__"],
    description=short_descr,
    long_description=readme + '\n\n' + history,
    author="ilhemghoul, ",
    author_email="ilhem.el.ghoul@gmail.com, ",
    url='https://github.com/VirtualPlants/segmentation_evaluation',
    license='cecill-c',
    zip_safe=False,

    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        ],
    tests_require=[
        "mock",
        "nose",
        ],
    entry_points={},
    keywords='',
    test_suite='nose.collector',
)
# #}
# change setup_kwds below before the next pkglts tag

# do not change things below
# {# pkglts, pysetup.call
setup(**setup_kwds)
# #}
