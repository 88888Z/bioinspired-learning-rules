# Bio-Inspired Learning Rules: Backpropagation vs. Feedback Alignment on MNIST

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A systematic comparison of **Backpropagation (BP)** and **Feedback Alignment (FA)** — a biologically plausible learning rule — on the MNIST digit classification task.

**Team NeuroAligners** — Neuromatch Academy Microlearning Project (Summer 2025)  
*Team Lead: [Alireza AlaeiFaradonbeh](https://github.com/alirezaalaeifaradonbeh)*

---

## Overview

Backpropagation (BP) drives modern deep learning but requires symmetric weight transport — a biologically implausible mechanism. Feedback Alignment (FA) bypasses this by using **fixed random feedback weights**, offering a more brain-like alternative (Lillicrap et al., 2016).

This project investigates:
- **Performance**: Accuracy, convergence speed, noise robustness
- **Learning dynamics**: Weight update alignment (cosine similarity), signal-to-noise ratio (SNR)
- **Representations**: Latent feature similarity, weight norm evolution

## Key Results

### Clean Test Accuracy (5 Epochs)

| Metric | Backpropagation | Feedback Alignment |
|--------|:--------------:|:-----------------:|
| **Test Accuracy** | **97.0%** | 83.5% |
| Test Loss | 0.107 | 0.685 |
| Epochs to 90% Val Acc | **1** | Not reached |

### Noise Robustness

FA demonstrates **more graceful degradation** under extreme noise:

| Gaussian Noise σ | BP Acc | FA Acc |
|:----------------:|:-----:|:------:|
| 0.0 | **97.0%** | 83.5% |
| 1.0 | 66.2% | **57.2%** |
| 2.0 | 32.2% | **33.9%** |
| 3.0 | 21.4% | **25.6%** |

At σ ≥ 1.8, FA **matches or exceeds BP**, suggesting its less specialized representations are more robust to severe input corruption.

## Repository Structure

```
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
├── notebook/
│   └── main.ipynb               ← Full experiment notebook (runnable)
├── proposal/
│   ├── NeuroAI-Final Proposal.docx
│   └── README.md                ← Proposal summary
├── poster/
│   └── NeuroAligners-Poster.pptx
├── report/
│   ├── Feedback Alignment in Deep Networks.md   ← Polished final report
│   ├── Feedback Alignment in Deep Networks.docx
│   ├── METHODS & ARCHITECTURE.docx
│   ├── project code + Progress Report.docx
│   └── workflow of code.docx
├── references/
│   ├── Papers.docx              ← Annotated bibliography
│   ├── README.md                ← Reference paper list
│   └── *.pdf                    ← Key reference papers
├── results/
│   └── NeuroMatch_Result.zip    ← All experiment outputs (252 files)
└── team/
    └── members.md               ← Team roster with roles & contributions
```

## Getting Started

```bash
git clone https://github.com/88888Z/bioinspired-learning-rules.git
cd bioinspired-learning-rules
pip install -r requirements.txt
jupyter notebook notebook/main.ipynb
```

### Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- torchvision, matplotlib, seaborn, numpy

## Results

The notebook generates 6 key plots covering accuracy/loss curves, noise robustness (Gaussian & salt-and-pepper), per-layer bias/SNR analysis, latent representation similarity, weight norms, and batch-level dynamics. Archived results for 9 hyperparameter configurations are in `results/`.

## Team

| Role | Name |
|------|------|
| **Team Lead & Lead Developer** | **Alireza AlaeiFaradonbeh** |
| Core Implementation | Morteza Hashemi |
| Data & Architecture | Yasaman Malahzadeh |
| Research & Literature | Avisa Fallah |
| Progress & Reporting | Melina Mostafa |
| Literature | Mobina Golsefatan |
| Literature | Sina Pirmoradian |

See [`team/members.md`](team/members.md) for full details.

## References

- Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support error backpropagation for deep learning. *Nature Communications*, 7, 13276.
- Konishi & Miura (2024). Biologically Plausible CNNs and Autoencoders with Feedback Alignment.
- LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- Nayebi, A., et al. (2020). Identifying learning rules from neural network observables. *NeurIPS*.
