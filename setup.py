# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="unlp",
    version="1.4.1",
    author="hanscal",
    author_email="hanscalcai@163.com, unagi_jason@163.com",
    description="a library that get text embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hanscal/unlp",
    packages=setuptools.find_packages(),
    package_data = {'': ['*.txt'],},
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
)
