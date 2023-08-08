from setuptools import setup, find_packages

setup(
    name='wzyFunc',
    version='0.1',
    author='Ziyang Wang',
    author_email='ziyangw@yeah.net',
    description='My Python Package',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'matplotlib>=3.1.3',
        'statsmodels>=0.11.0',
        'seaborn>=0.10.0',
        'scipy>=1.4.1',
        'scikit-learn>=0.22.1',
        'xgboost>=1.0.2',
        'lightgbm>=2.3.1',
        'torch>=1.4.0',
        'torchvision>=0.5.0',
        'tqdm>=4.42.1',
        'pyecharts>=1.7.1'
    ]
)