# Workflow: Comparing Backpropagation and Feedback Alignment

**Author:** Alireza AlaeiFaradonbeh (Team Lead)

---

## 1. Core Concept

This project trains two identical feedforward networks on MNIST — one with **Backpropagation (BP)**, the other with **Feedback Alignment (FA)** — and compares them across performance, learning dynamics, and robustness metrics.

- **BP** uses exact gradient signals computed via the chain rule (standard deep learning)
- **FA** replaces exact backward weight matrices with **fixed random feedback weights**, inspired by biological learning (Lillicrap et al., 2016)

## 2. Code Architecture

### 2.1 Model Definition

```python
class FFN(nn.Module):
    # 784 -> 128 (ReLU) -> 10
    # Two identical instances: m_bp, m_fa (same seed)
```

### 2.2 Update Rules

**Backpropagation:**
- Standard forward pass → cross-entropy loss → `loss.backward()` → optimizer step (Adam, lr=1×10⁻³)

**Feedback Alignment:**
- Forward pass through first linear layer → ReLU → final linear → logits
- Compute output error: one-hot(target) − softmax(logits)
- Back-project through **fixed random matrix B** (not transposed weights)
- Apply ReLU derivative mask on hidden pre-activations
- Compute local weight/bias updates using local activations
- Apply updates in-place (no autograd)

### 2.3 Training Loop

```
for epoch in 1..EPOCHS:
    for each batch:
        1. BP update on m_bp → record loss & learning rate
        2. FA update on m_fa → record loss
        3. For first 40 batches:
           - Save weights before update
           - Compute update vectors (Δw = w_after − w_before)
           - Accumulate for bias (cosine similarity) & SNR computation
    After epoch:
        - Evaluate both models on train/val sets
        - Compute per-layer bias & SNR
        - Track latent representation similarity
        - Track weight norms
```

### 2.4 Metrics

| Metric | Implementation |
|--------|---------------|
| Accuracy & Loss | Standard cross-entropy, argmax prediction |
| Convergence | Validation loss slope (first 3 epochs), epochs to 90% accuracy |
| Noise Robustness | Accuracy degradation under Gaussian (σ ∈ [0, 3]) and salt-and-pepper (α ∈ [0, 0.8]) noise |
| Update Alignment | Cosine similarity between mean weight updates of BP and FA |
| Signal-to-Noise Ratio | ‖μ‖² / ΣVar for each algorithm's weight updates |
| Latent Similarity | Cosine similarity between hidden-layer activations (BP vs FA on same inputs) |
| Weight Norms | Frobenius norm of each weight matrix per epoch |

## 3. Key Findings

1. **BP achieves higher accuracy** (97.0% vs 83.5% on test set) and converges faster
2. **FA shows competitive noise robustness** — at high noise levels (σ ≥ 1.8), FA matches or exceeds BP
3. **Weight updates show moderate alignment** (cosine similarity ~0.6–0.8) that improves over training
4. **BP updates are cleaner** (higher SNR), explaining its faster convergence
5. **Internal representations converge** — hidden-layer activations become increasingly similar over time

## 4. References

- Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support error backpropagation for deep learning. *Nature Communications*, 7, 13276.
- Konishi & Miura (2024). Biologically Plausible CNNs and Autoencoders with Feedback Alignment.
