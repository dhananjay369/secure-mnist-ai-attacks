# Dataset Instructions

This project uses the MNIST handwritten digit dataset.

## Dataset Download

The MNIST dataset is downloaded automatically by the project scripts when required, provided an internet connection is available.

## Generate the Poisoned Dataset

From the repository root, activate the virtual environment and run:

    python -m attacks.poisoning_attack

## Why Dataset Files Are Not Included

Raw and generated dataset files are excluded from Git to keep the repository lightweight. Run the scripts to download or regenerate them locally.
