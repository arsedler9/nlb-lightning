# Runs the installation. See the following for more detail:
# https://docs.python.org/3/distutils/setupscript.html

from setuptools import find_packages, setup

# Avoids duplication of requirements
with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setup(
    name="nlb-lightning",
    author="Andrew R. Sedler",
    author_email="arsedler9@gmail.com",
    description="PyTorch Lightning utilities that make it easier to train and evaluate "
        "deep models for the Neural Latents Benchmark.",
    url="https://github.com/arsedler9/nlb-lightning",
    install_requires=requirements,
    packages=find_packages(),
)
