# 📊 EmGen: An Empirical Study of Generative Models

# 🛠️ Installation Guide

## 📦 Prerequisites

Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed on your system.

You can check by running:

```bash
conda --version
```

## 🚀 Environment Setup
```bash
git clone git@github.com:rashikshrestha/emgen.git
cd emgen
conda env create -f environment.yml
echo "export PYTHONPATH=$(pwd):\$PYTHONPATH" >> ~/.bashrc # use zshrc if you are using zsh terminal
source ~/.bashrc
conda activate emgen
```

# References
- [Tiny Diffusion](https://github.com/tanelp/tiny-diffusion)