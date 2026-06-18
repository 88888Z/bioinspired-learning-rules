# Project Proposal

**NeuroAI-Final Proposal.docx** — The official project proposal submitted for Neuromatch Academy.

### Research Questions

1. **Q3**: How do performance and robustness of FA compare to BP? (convergence, accuracy, noise robustness)
2. **Q4**: What are the fundamental differences in learning dynamics? (update alignment, SNR)
3. **Q7**: To what extent does FA approximate the global gradient descent direction defined by BP?

### Proposed Methodology

- MLP with 128 hidden units, ReLU, trained on MNIST
- Compare BP (Adam) vs FA (fixed random feedback, manual SGD)
- Evaluate accuracy, loss, cosine similarity, SNR, noise robustness
