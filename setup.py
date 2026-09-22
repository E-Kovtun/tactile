#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE.md file in the root directory of this source tree.
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="tactile_ssl",
    version="0.0.1",
    description="Tactile-JEPA: topology-aware self-supervised learning for tactile sensors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="CC-BY-NC-4.0",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
)
