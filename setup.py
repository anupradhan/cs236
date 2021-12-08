from setuptools import setup, find_packages

setup(
    name='pytorch_cgan',
    version='1.0.0',
    description='Text Conditioned Image Generation',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch==1.10.0",
        "numpy==1.17",
        "boto3",
        "torchvision~=0.11.1",
        "torchaudio",
        "h5pickle==0.4",
        "matplotlib==3.3.0",
        "wget"
    ],
)