# PReF: Language Model Personalization via Reward Factorization

This repository contains the official implementation of the paper [Language Model Personalization via Reward Factorization](https://arxiv.org/abs/2503.06358).

## Abstract

Modern large language models (LLMs) are optimized for human-aligned responses using Reinforcement Learning from Human Feedback (RLHF). However, existing RLHF approaches assume a universal preference model and fail to account for individual user preferences, limiting their effectiveness in personalized applications. We introduce a framework that extends RLHF to enable user personalization by leveraging the assumption that user preferences lie in a low-dimensional space. Instead of training a separate model per user, we represent user-specific rewards as a linear combination of base reward functions. Using only ~10 user responses, our method can infer user-specific rewards and align LLM outputs accordingly. We validate our approach through experiments with both synthetic and real users, demonstrating significant personalization achieved by our method. In human evaluations, our method achieves a 67% win rate over default GPT-4o responses.

## Setup

Create a conda environment:
```bash
conda create -n pref python=3.10
```

Activate the environment:
```bash
conda activate pref
```

Download the data from [here](https://drive.google.com/file/d/1QSeUKoqml8VVQvmPYUFo9C2PxvKfgfjy/view?usp=sharing) and extract it into the `data/` directory.

## Citation

If you find this work useful, please cite our paper:
```bibtex
@article{pref2024,
title={Language Model Personalization via Reward Factorization},
author={[Authors]},
journal={arXiv preprint arXiv:2503.06358},
year={2024}
}
```