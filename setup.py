"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gabor_fit',
    description='Fit dictionary to gabor filters.',
    long_description=long_description,
    author='Jesse Livezey',
    author_email='jesse.livezey@berkeley.edu'
)
"""
# To provide executable scripts, use entry points in preference to the
# "scripts" keyword. Entry points provide cross-platform support and allow
# pip to create the appropriate form of executable for the target platform.
entry_points={
    'console_scripts': [
        'ecog=bin:main',
    ],
},
"""
