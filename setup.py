import triangle

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


setup(
    name="triangle.py",
    version=triangle.__version__,
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    py_modules=["triangle"],
    description="MOAR TRIANGLEZ",
    long_description=open("README").read(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
