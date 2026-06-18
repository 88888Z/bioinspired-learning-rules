# Feedback Alignment in Deep Networks: Learning Dynamics, Robustness, and Biological Plausibility

**Team NeuroAligners** — Neuromatch Academy Microlearning Project

> *Team Lead: Alireza AlaeiFaradonbeh*

---

## Abstract

Backpropagation (BP) is the foundation of modern deep learning, yet its requirement for symmetric weight transport makes it biologically implausible. Feedback Alignment (FA) offers a more brain-like alternative by using fixed, random feedback weights to propagate error signals. This project systematically compares BP and FA on the MNIST digit classification task using identical feedforward architectures. We evaluate accuracy, convergence speed, noise robustness, weight update alignment (cosine similarity), signal-to-noise ratio (SNR), latent representation similarity, and weight norm dynamics. Our results confirm that while BP achieves higher clean accuracy (97.0% vs. 83.5% on the test set) and faster convergence, FA demonstrates competitive learning and exhibits more graceful degradation under high input noise — suggesting potential advantages for noisy, real-world environments.

---

## 1. Introduction

The backpropagation algorithm (Rumelhart et al., 1986) remains the dominant method for training deep neural networks. However, its reliance on precise symmetric feedback connections — the "weight transport problem" (Grossberg, 1987) — is widely considered biologically unrealistic. The brain, with its messy, asymmetric synaptic connections, clearly learns without such a mechanism.

Feedback Alignment (Lillicrap et al., 2016) addresses this by replacing the transposed forward weight matrices used in backpropagation with fixed, random feedback matrices. Despite receiving only approximate error signals, networks trained with FA have been shown to learn meaningful representations across a range of tasks (Lillicrap et al., 2016; Konishi & Miura, 2024).

This project investigates three core questions:
1. **Performance**: How do FA and BP compare in accuracy, convergence speed, and noise robustness?
2. **Learning Dynamics**: How aligned are the weight updates of FA with those of BP, and how does update quality (SNR) differ?
3. **Representational Similarity**: Do FA and BP learn comparable internal representations despite their different update rules?

---

## 2. Methodology

### 2.1 Model Architecture

We use a feedforward neural network (FFN) with:
- **Input**: 784 units (28×28 MNIST image, flattened)
- **Hidden layer**: 128 units, ReLU activation
- **Output**: 10 units (digit classes 0–9)

Two identical networks are instantiated with the same random seed for fair comparison: one trained with BP, the other with FA.

### 2.2 Dataset

- **MNIST** (LeCun et al., 1998): 70,000 grayscale handwritten digit images
- **Split**: 24,000 training / 6,000 validation / 10,000 test
- **Preprocessing**: Normalized to mean = 0.1307, std = 0.3081
- **Batch size**: 128

### 2.3 Training Protocols

| Parameter | Backpropagation | Feedback Alignment |
|-----------|:---------------:|:------------------:|
| Optimizer | Adam (lr = 1×10⁻³) | Manual SGD (lr = 1×10⁻³) |
| Loss | Cross-entropy | Cross-entropy |
| Epochs | 5 | 5 |
| Feedback weights | Transposed forward weights | Fixed random matrix ~𝒩(0, 0.1) |
| Weight update | `loss.backward()` + optimizer step | Local gradient through fixed B |

### 2.4 Evaluation Metrics

- **Accuracy & Loss**: Training, validation, and test sets
- **Convergence**: Validation loss slope (first 3 epochs), epochs to 90% accuracy
- **Noise Robustness**: Gaussian noise (σ ∈ [0, 3.0]) and salt-and-pepper noise (α ∈ [0, 0.8])
- **Update Alignment**: Cosine similarity between mean weight updates of BP and FA
- **Signal-to-Noise Ratio (SNR)**: ‖μ‖² / ΣVar for weight updates of each algorithm
- **Latent Similarity**: Cosine similarity between hidden-layer activations of BP and FA
- **Weight Norms**: Frobenius norm of weight matrices per epoch

---

## 3. Results

### 3.1 Accuracy and Convergence

| Metric | Backpropagation | Feedback Alignment |
|--------|:---------------:|:------------------:|
| Test Accuracy | **97.0%** | 83.5% |
| Test Loss | **0.107** | 0.685 |
| Val Accuracy | **96.4%** | 82.1% |
| Epochs to 90% Val Acc | **1** | Not reached |
| Val Loss Slope (early) | −0.051 | −0.312 |

BP converges rapidly, reaching 90% validation accuracy within a single epoch. FA learns more gradually and plateaus at a lower peak accuracy, though it continues to improve over the training window.

### 3.2 Noise Robustness

Under Gaussian noise, FA shows **more graceful degradation** at extreme noise levels despite lower clean accuracy:

| Gaussian Noise σ | BP Accuracy | FA Accuracy |
|:----------------:|:----------:|:-----------:|
| 0.0 | **97.0%** | 83.5% |
| 0.5 | 93.5% | **78.3%** |
| 1.0 | 66.2% | **57.2%** |
| 1.5 | 48.0% | **45.4%** |
| 2.0 | 32.2% | **33.9%** |
| 2.5 | 24.9% | **28.4%** |
| 3.0 | 21.4% | **25.6%** |

At σ ≥ 1.8, FA **matches or exceeds BP** in accuracy. This suggests that FA's less specialized, noisier learning signal produces representations that are less brittle to severe input corruption — a potential advantage for real-world noisy environments.

### 3.3 Weight Update Alignment

The cosine similarity between BP and FA weight updates ranges from **0.6–0.8**, indicating moderate directional alignment. Alignment improves as training progresses, echoing findings by Lillicrap et al. (2016) that FA updates become more correlated with the true gradient over time.

### 3.4 Signal-to-Noise Ratio

BP consistently exhibits **higher SNR** (cleaner, more stable updates) across all layers. FA shows lower SNR, reflecting the inherent noise in its random feedback mechanism. This explains FA's slower convergence and higher final loss.

### 3.5 Latent Representation Similarity

Despite different update rules, the hidden-layer representations of BP and FA networks converge over training, reaching a cosine similarity of ~0.5–0.6. This indicates that FA discovers broadly similar internal features to BP, even with imprecise gradient information.

---

## 4. Discussion

### 4.1 Trade-offs Between BP and FA

| Aspect | BP Advantage | FA Advantage |
|--------|:------------:|:------------:|
| Clean accuracy | ✓ Significantly higher | — |
| Convergence speed | ✓ Much faster | — |
| High-noise robustness | — | ✓ More graceful decay |
| Biological plausibility | ✗ (weight transport) | ✓ (local, asymmetric) |
| Update stability | ✓ (high SNR) | — |

### 4.2 Why FA Excels Under High Noise

FA's reliance on random feedback weights may act as an implicit regularizer, preventing overfitting to input statistics. The noisier update signals lead to broader optima that are more robust to severe input corruption — analogous to how stochastic noise in SGD can improve generalization.

### 4.3 Limitations

- Single architecture tested (784→128→10)
- Limited to 5 epochs due to computational constraints
- Only one dataset (MNIST)
- Basic FA only; advanced variants (DFA, PFA) not explored

---

## 5. Future Work

- Extend training to 100+ epochs to study overfitting dynamics
- Test on wider/deeper networks and harder datasets (Fashion-MNIST, CIFAR-10)
- Implement DFA, PFA, and FDFA for comparative analysis
- Explore noise pretraining (Cheon et al., 2024) to improve FA alignment
- Analyze temporal evolution of latent features using CKA

---

## 6. Conclusion

Backpropagation remains superior in clean accuracy and convergence speed, achieving 97.0% test accuracy on MNIST versus 83.5% for Feedback Alignment. However, FA demonstrates that effective learning is possible without symmetric weight transport — a key step toward biologically plausible AI. Moreover, FA's more graceful degradation under extreme noise suggests potential advantages for deployment in noisy, real-world environments. These findings contribute to our understanding of the trade-offs involved in moving from artificial to brain-inspired learning algorithms.

---

## References

1. Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2016). Random synaptic feedback weights support error backpropagation for deep learning. *Nature Communications*, 7, 13276.
2. Konishi & Miura (2024). Biologically Plausible CNNs and Autoencoders with Feedback Alignment. *SCML*.
3. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
4. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
5. Grossberg, S. (1987). Competitive learning: From interactive activation to adaptive resonance. *Cognitive Science*, 11(1), 23–63.
6. Cheon, J., Lee, S. W., & Paik, S.-B. (2024). Pretraining with Random Noise for Fast and Robust FA. *NeurIPS*.
7. Nayebi, A., et al. (2020). Identifying learning rules from neural network observables. *NeurIPS*.
