import os
import pathlib
import platform

# import pkg_resources
from setuptools import find_packages, setup

# Basic Information
NAME = "unicorrn"
VERSION = "0.1.0"
DESCRIPTION = "A python package for UniCorrn: Unified Correspondence Transformer across Images and Point Clouds 2D and 3D"

# Define the classifiers
# See https://pypi.python.org/pypi?%3Aaction=list_classifiers
# CLASSIFIERS = [
#     "Development Status :: 2 - Pre-Alpha",
#     "Intended Audience :: Education",
#     "Natural Language :: English",
#     "Programming Language :: Python :: 3.10",
# ]

# Define the keywords
# KEYWORDS = [
#     "pytorch",
#     "machine learning",
#     "deep learning",
#     "image inpainting",
#     "computer vision",
# ]

# Directories to ignore in find_packages
EXCLUDES = ()

# Important Paths
PROJECT = os.path.abspath(os.path.dirname(__file__))

REQUIRE_PATH = "requirements.txt"

# with pathlib.Path(REQUIRE_PATH).open() as requirements_txt:
#     INSTALL_REQUIRES = [
#         str(requirement)
#         for requirement in pkg_resources.parse_requirements(requirements_txt)
#     ]


CONFIG = {
    "name": NAME,
    "version": VERSION,
    "description": DESCRIPTION,
    "author": "Prajnan Goswami",
    # "classifiers": CLASSIFIERS,
    # "keywords": KEYWORDS,
    # "url": REPOSITORY,
    "packages": find_packages(
        where=PROJECT, include=["unicorrn", "unicorrn.*"], exclude=EXCLUDES
    ),
    # "install_requires": INSTALL_REQUIRES,
    "python_requires": ">=3.10",
    "test_suite": "tests",
    "tests_require": ["pytest>=3"],
    "include_package_data": True,
}

if __name__ == "__main__":
    setup(**CONFIG)
