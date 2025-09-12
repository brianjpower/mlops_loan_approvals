from setuptools import setup, find_packages

setup(
    name="loan_mlops",
    version="0.0.1",
    author="Brian Power",
    description="A ML package to predict loan outcomes from certain financial data",
    packages=find_packages(),
    classifiers=[
        "Programming Lanuage :: Python :: 3",
        "Licence :: OSI approved :: bplic",
        "Operating System :: OS independent",
    ],
)