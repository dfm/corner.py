#!/usr/bin/env python

import re
import os
import sys
from setuptools import setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

# Hackishly synchronize the version.
version = re.findall(r"__version__ = \"(.*?)\"", open("corner.py").read())[0]


setup(
    name="corner",
    version=version,
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    url="https://github.com/dfm/corner.py",
    py_modules=["corner"],
    description="Make some beautiful corner plots of samples.",
    long_description=open("README.rst").read(),
    package_data={"": ["LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
