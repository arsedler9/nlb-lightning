# nlb-lightning
PyTorch Lightning utilities that make it easier to train and evaluate deep models for the Neural Latents Benchmark.

# Installation
Clone the entire `nlb-lightning` repo, including submodules. Then, create and activate a `conda` environment and install the `nlb_tools` and `nlb_lightning` packages.
```
git clone --recurse-submodules git@github.com:arsedler9/nlb-lightning.git
cd nlb-lightning
conda create --name nlb-lightning python=3.9
conda activate nlb-lightning
pip install -e nlb_tools
pip install -e .
```
