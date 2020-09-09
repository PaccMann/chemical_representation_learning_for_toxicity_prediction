"""Install package."""
from setuptools import setup, find_packages

setup(
    name='toxsmi',
    version='0.0.1',
    description=(
        'PyTorch implementation of toxicity prediction models from SMILES.'
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PaccMann/toxsmi',
    author='Jannis Born, Greta Markert, Matteo Manica',
    author_email=(
        'jab@zurich.ibm.com, greta.markert@gmail.com, drugilsberg@gmail.com'
    ),
    install_requires=[
        'numpy', 'torch', 'six', 'scikit-learn', 'tensorflow<2.0',
        'deepchem==2.2.1.dev54', 'Pillow==7.0.0',
        'paccmann_predictor @ git+https://github.com/PaccMann/paccmann_predictor@latest',
        'pytoda @ git+https://github.com/PaccMann/paccmann_datasets@0.1.0   '
    ],
    packages=find_packages('.'),
    zip_safe=False
)
