#!/usr/bin/env python

import os
import sys
from setuptools import setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist bdist_wheel upload")
    sys.exit()

# Hackishly inject a constant into builtins to enable importing of the
# package before the dependencies are installed.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__CORNER_SETUP__ = True
import corner  # NOQA


setup(
    name="corner",
    version=corner.__version__,
    author="Daniel Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/corner.py",
    packages=["corner"],
    description="Make some beautiful corner plots of samples.",
    long_description=open("README.rst").read(),
    package_data={"": ["README.rst", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    install_requires=["numpy", "matplotlib"],
)
