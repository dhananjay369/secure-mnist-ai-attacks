# Secure MNIST AI Attacks

This repository contains red-team experiments for an MNIST
handwritten digit classification model.

## Experiments

- FGSM adversarial example generation and evaluation
- Label-flipping data poisoning
- Training and evaluation of a poisoned model
- Adversarial training
- Baseline-versus-defended model comparison

## Project Structure

- attacks/: attack generation and evaluation scripts
- src/: CNN implementation and shared utilities
- defense/: adversarial training implementation

## Requirements

Use Python 3.11 and install the project dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt