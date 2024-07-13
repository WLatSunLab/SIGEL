from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='sigel',
    version='1.0.0',
    description='A context-aware genomic language model for gene spatial expression imputation, pattern detection, and function discovery',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Wenlin Li',
    author_email='zipging@gmail.com',
    url='https://github.com/WLatSunLab/SIGEL',
    license='MIT license',
    python_requires=">=3.9",
    install_requires=[
        'torch>=1.13.0',
        'rpy2>=1.13.0',
        'scikit-learn>=1.2.0',
        'scanpy>=1.9.6',
        'scipy>=1.11.4',
        'pandas>=1.5.2',
        'numpy>=1.21.6',
        'sympy>=1.11.1',
        'anndata>=0.10.3',
        'SpaGCN>=1.2.7',
        'tqdm>=4.64.1'
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    zip_safe=False,
)
