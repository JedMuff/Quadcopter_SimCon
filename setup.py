#!/usr/bin/env python3
"""
Setup script for Quadcopter Simulation and Control Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Quadcopter Simulation and Control Framework"

# Read requirements from requirements.txt if it exists
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Fallback requirements based on the imports found in the codebase
FALLBACK_REQUIREMENTS = [
    'numpy>=1.20.0',
    'matplotlib>=3.3.0',
    'sympy>=1.8.0',
    'scipy>=1.7.0',
]

setup(
    name="quadcopter-simcon",
    version="1.0.0",
    author="John Bass (Original), Enhanced Framework",
    author_email="john.bobzwik@gmail.com",
    description="A comprehensive quadcopter simulation and control framework with configurable drone types",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/bobzwik/Quadcopter_SimCon",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements() or FALLBACK_REQUIREMENTS,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'flake8',
            'mypy',
        ],
        'docs': [
            'sphinx',
            'sphinx-rtd-theme',
        ],
    },
    entry_points={
        'console_scripts': [
            'drone-sim=examples.run_3D_simulation_configurable:main',
        ],
    },
    package_data={
        'drone_sim': [
            'configs/*.json',
        ],
    },
    include_package_data=True,
    keywords="quadcopter simulation control robotics drone flight dynamics",
    project_urls={
        "Bug Reports": "https://github.com/bobzwik/Quadcopter_SimCon/issues",
        "Source": "https://github.com/bobzwik/Quadcopter_SimCon",
        "Documentation": "https://github.com/bobzwik/Quadcopter_SimCon/wiki",
    },
)