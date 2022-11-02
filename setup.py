from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup_args = dict(
    name='ecglib',
    version='1.0.1',
    description='ECG library with pretrained models and tools for ECG analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache License 2.0',
    packages=find_packages(),
    include_package_data=True,
    author='Aram Avetisyan',
    author_email='a.a.avetisyan@gmail.com',
    install_requires=['pandas',
        'ecg_plot',
        'tqdm',
        'numpy',
        'torch',
        'scipy',
        'ipywidgets',
        'pyyaml',
        'PyWavelets',
        'wfdb',
        'hydra-core',
        'omegaconf',
    ],
    keywords=['ecg analysis', 'pytorch', 'pretrained models', 'ecg preprocessing', 'ecg datasets'],
    url = 'https://github.com/ispras/EcgLib',
    python_requires='>=3.6',
)


if __name__ == '__main__':
    setup(**setup_args)
