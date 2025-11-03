#!/usr/bin/env python3
"""
Setup script for corridor-nav
"""

from setuptools import setup, find_packages

# Read the README file for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="corridor-nav",
    version="0.1.0",
    author="Stephen Monnet",
    author_email="stephen.monnet@outlook.com",
    description="Corridor navigation optimization for marine vessels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stiefen1/corridor-nav",
    project_urls={
        "Bug Tracker": "https://github.com/stiefen1/corridor-nav/issues",
        "Repository": "https://github.com/stiefen1/corridor-nav",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "shapely>=1.8.0",
        "cartopy>=0.20.0",
        "geopandas>=0.10.0",
        "pyproj>=3.2.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
        ],
        "jupyter": [
            "jupyter",
            "ipykernel",
            "matplotlib-inline",
        ],
        "gui": [
            "customtkinter>=5.0.0",
            "pygame>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add command-line scripts here if needed
            # "corridor-nav=corridor_nav.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)