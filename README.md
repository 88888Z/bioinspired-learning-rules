# Bio-inspired Learning Rules: Backpropagation vs Feedback Alignment on MNIST

A computational experiment comparing **Backpropagation (BP)** — the standard neural network training algorithm — against **Feedback Alignment (FA)**, a biologically-plausible alternative that uses fixed random feedback weights instead of symmetric backward connections.

## Overview

Standard backpropagation requires symmetric forward/backward weights, which is biologically implausible. Feedback Alignment (Lillicrap et al., 2016) sidesteps this by propagating errors through fixed random matrices. This project empirically compares both algorithms on MNIST digit classification, analyzing:

- Accuracy & convergence speed
- Weight update alignment (bias) and signal-to-noise ratio (SNR)
- Noise robustness (Gaussian and salt-and-pepper noise)
- Latent representation similarity
- Weight norm dynamics

## Key Results

### Clean Accuracy (8 epochs)

| Metric | Backpropagation | Feedback Alignment |
|--------|:--------------:|:-----------------:|
| Test Accuracy | **97.0%** | 83.5% |
| Epochs to 90% Val Acc | **1** | never reached |
| Val Loss Slope (early) | −0.05 | −0.31 |

### Noise Robustness

At high noise levels, FA shows **comparable or better robustness** despite lower clean accuracy:

| Gaussian Noise σ | BP Acc | FA Acc |
|:---------------:|:-----:|:-----:|
| 0.0 | **97.0%** | 83.5% |
| 1.0 | 66.2% | **57.2%** |
| 2.0 | 32.2% | **33.9%** |
| 3.0 | 21.4% | **25.6%** |

FA degrades more gracefully, suggesting its less specialized representations are less brittle to input corruption.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- seaborn
- numpy

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Run the notebook end-to-end:

```bash
jupyter notebook bioinspired-learning-rule_microlearning.ipynb
```

Or execute as a script:

```bash
jupyter nbconvert --to script bioinspired-learning-rule_microlearning.ipynb --stdout | python
```

## Results

The notebook generates 6 plots covering:
1. **Plot 1** — Train/validation accuracy & loss curves
2. **Plot 2** — Noise robustness curves (Gaussian & salt-and-pepper)
3. **Plot 3** — Bias (cosine similarity) & SNR of weight updates per layer
4. **Plot 4** — Latent representation similarity between BP and FA
5. **Plot 5** — Weight norms over epochs
6. **Plot 6** — Batch-level loss & learning rate dynamics

## References

- Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2016). Random synaptic feedback weights support error backpropagation for deep learning. *Nature Communications*, 7, 13276.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
