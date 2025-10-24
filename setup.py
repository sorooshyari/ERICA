"""Setup configuration for ERICA package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='erica-clustering',
    version='0.1.3',
    author='Siamak Sorooshyari, Shawn Shirazi',
    author_email='shawn.shirazi@example.com',
    description='ERICA - A robust clustering replicability analysis tool using iterative clustering assignments',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/shawnshirazi/ERICA_PyPI',
    project_urls={
        'Bug Tracker': 'https://github.com/shawnshirazi/ERICA_PyPI/issues',
        'Documentation': 'https://github.com/shawnshirazi/ERICA_PyPI/blob/main/docs/',
        'Source Code': 'https://github.com/shawnshirazi/ERICA_PyPI',
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'examples', 'examples.*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    keywords='clustering replicability stability machine-learning bioinformatics monte-carlo',
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'pyyaml>=6.0',
    ],
    extras_require={
        'plots': [
            'plotly>=5.0.0',
            'matplotlib>=3.5.0',
        ],
        #'gui': [
        #    'gradio>=4.0.0',
        #],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=0.990',
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
        'all': [
            'plotly>=5.0.0',
            'matplotlib>=3.5.0',
            #'gradio>=4.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'erica=erica.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

