# Session 02: Modern Approaches

From handwritten digits to frontier LLM techniques — what modern AI can do and where research is heading.

**Demos:**
- `uv run sand session02-mnist` — CNN on MNIST handwritten digits
- `uv run sand session02-custom` — feedforward network on Iris dataset

---

## MNIST Model

<details>
<summary><strong>What Is MNIST and Why Does It Matter?</strong></summary>

MNIST (Modified National Institute of Standards and Technology) is a dataset of 70,000 grayscale images of handwritten digits (0–9), each 28×28 pixels. Created in 1998 by Yann LeCun, Corinna Cortes, and Christopher Burges, it became the standard benchmark for image classification.

Why it matters pedagogically:
- **Small enough to train on a laptop** in seconds to minutes
- **Complex enough to need real techniques** — a linear model gets ~92%, but you need a CNN to break 99%
- **Universally understood** — everyone knows what the digits 0–9 look like

MNIST is the "hello world" of deep learning. Every major architecture has been tested on it.

</details>

<details>
<summary><strong>CNN Architecture</strong></summary>

Our model uses a convolutional neural network:

```
Input: 1×28×28 (grayscale image)
    ↓
Conv2d(1→16, 3×3, padding=1) → ReLU → MaxPool(2×2)    → 16×14×14
    ↓
Conv2d(16→32, 3×3, padding=1) → ReLU → MaxPool(2×2)   → 32×7×7
    ↓
Flatten                                                  → 1568
    ↓
Linear(1568→128) → ReLU                                 → 128
    ↓
Linear(128→10)                                           → 10 class scores
```

**Convolutional layers** learn spatial feature detectors — edges, curves, corners in the first layer; combinations of those features (loops, strokes, intersections) in the second.

**Max pooling** reduces spatial dimensions by taking the maximum value in each 2×2 region, providing translation invariance (a digit shifted slightly still activates the same detectors).

</details>

<details>
<summary><strong>Training Results</strong></summary>

In just 3 epochs (one pass through all 60,000 training images = one epoch), the CNN typically achieves **~98–99% test accuracy** on the 10,000 test images.

This demonstrates:
- CNNs are remarkably data-efficient for structured spatial data
- Modern hardware (GPU/MPS) trains this in seconds
- The gap from ~92% (linear model) to ~99% (CNN) is the value of hierarchical feature learning

</details>

---

## Custom Model — Iris Dataset

<details>
<summary><strong>Applying Neural Networks to Tabular Data</strong></summary>

Not all data is images. The Iris dataset contains 150 samples of three flower species, described by four measurements: sepal length, sepal width, petal length, and petal width.

Our model is a simple feedforward network:

```
Input: 4 features
    ↓
Linear(4→16) → ReLU
    ↓
Linear(16→3) → CrossEntropyLoss
```

This demonstrates that the same train/predict pattern from MNIST applies to any numeric data. The architecture changes (no convolutions needed for tabular data), but the training loop is identical:

1. Forward pass
2. Compute loss
3. Backward pass (compute gradients)
4. Update weights

</details>

<details>
<summary><strong>Why Standardize Features?</strong></summary>

Before training, we standardize each feature to have mean=0 and standard deviation=1. This matters because:

- **Different scales confuse gradient descent:** Sepal length might range 4–8 cm while petal width ranges 0.1–2.5 cm. Without standardization, the larger-scale feature dominates gradient updates.
- **Faster convergence:** The loss surface becomes more spherical, so gradient descent takes more direct paths to the minimum.
- **Numerical stability:** Prevents very large or very small activations that can cause overflow or vanishing gradients.

scikit-learn's `StandardScaler` computes `(x - mean) / std` for each feature.

</details>

---

## Frontier LLM Capabilities

<details>
<summary><strong>Fine-Tuning: Full vs. LoRA</strong></summary>

**Fine-tuning** adapts a pre-trained model to a specific task by continuing training on task-specific data.

**Full fine-tuning** updates every parameter in the model. For a 7B-parameter model, this requires:
- ~28 GB of GPU memory just for model weights (FP32)
- Additional memory for gradients and optimizer states
- Risk of catastrophic forgetting (losing general capabilities)

**LoRA (Low-Rank Adaptation)** freezes the original weights and adds small trainable matrices to each layer:
- Instead of updating a weight matrix W directly, LoRA learns a low-rank decomposition: W + BA, where B and A are much smaller matrices
- A 7B model might need only ~10M trainable parameters (0.14% of the total)
- Reduces GPU memory by ~3× compared to full fine-tuning
- Multiple LoRA adapters can be hot-swapped at inference time

**When to use which:**
| Approach | Use When |
|----------|----------|
| Full fine-tuning | You have abundant compute, lots of data, and need maximum performance |
| LoRA | Limited compute, want to preserve general capabilities, need multiple task-specific adapters |
| No fine-tuning (prompting) | Your task can be described in natural language; the base model is capable enough |

**Paper:** Hu, E.J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.

</details>

<details>
<summary><strong>RAG (Retrieval-Augmented Generation)</strong></summary>

RAG combines a language model with a retrieval system to ground generation in external knowledge.

**Architecture:**

```
User Query
    ↓
Embedding Model → query vector
    ↓
Vector Database → retrieve top-K relevant documents
    ↓
LLM receives: [retrieved context] + [user query]
    ↓
Generated response (grounded in retrieved facts)
```

**Why RAG instead of fine-tuning?**
- **Up-to-date knowledge:** Swap documents without retraining
- **Verifiability:** You can see which documents the answer came from
- **Cost:** No GPU training required — just an embedding model and a vector store
- **Hallucination reduction:** The model generates from real documents rather than parametric memory

**When to use RAG vs. fine-tuning:**
| Approach | Best For |
|----------|----------|
| RAG | Factual QA, document search, knowledge that changes frequently |
| Fine-tuning | Style/tone adaptation, specialized reasoning, tasks requiring deep domain behavior |
| Both | Complex enterprise applications (fine-tune for domain behavior, RAG for current facts) |

**Paper:** Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.

</details>

---

## Other Modern Approaches

<details>
<summary><strong>Alpha Series (AlphaGo → AlphaZero → AlphaFold → AlphaEvolve)</strong></summary>

DeepMind's "Alpha" lineage demonstrates how reinforcement learning and neural networks can master domains previously thought to require human intuition.

**AlphaGo (2016):** Combined deep neural networks with Monte Carlo tree search to defeat world champion Lee Sedol at Go — a game with ~10^170 possible positions where brute-force search is impossible. Used human expert games for initial training, then improved through self-play.

**AlphaZero (2017):** Generalized AlphaGo's approach to chess, shogi, and Go using *only self-play* — no human games, no handcrafted features, no opening books. Starting from random play, it achieved superhuman performance in all three games within 24 hours.

**AlphaFold (2021):** Applied deep learning to protein structure prediction — determining a protein's 3D shape from its amino acid sequence. Won CASP14 (the field's benchmark competition) with accuracy competitive with experimental methods. The AlphaFold Protein Structure Database now contains predicted structures for nearly every known protein.

**AlphaEvolve (2025):** A coding agent that uses LLMs with evolutionary search to discover new algorithms. It found an improvement to matrix multiplication (the first advance over Strassen's algorithm in 56 years for 4×4 complex matrices) and has optimized critical Google infrastructure.

| System | Year | Domain | Key Innovation |
|--------|------|--------|---------------|
| AlphaGo | 2016 | Go | Deep RL + MCTS + human data |
| AlphaZero | 2017 | Chess/Shogi/Go | Pure self-play, no human data |
| AlphaFold | 2021 | Protein structure | Attention-based structure prediction |
| AlphaEvolve | 2025 | Algorithm discovery | LLM + evolutionary search |

</details>

<details>
<summary><strong>ENAS — Efficient Neural Architecture Search</strong></summary>

Instead of a human designing the network architecture, what if we let the AI design it?

Neural Architecture Search (NAS) uses a controller network to generate candidate architectures, trains them, and uses the validation performance as a reward signal to improve the controller. The original NAS paper (Zoph & Le, 2017) required 800 GPUs for 28 days.

**ENAS** (Pham et al., 2018) made this practical by sharing parameters across candidate architectures. Instead of training each candidate from scratch, all candidates share a single large network's weights, and the controller learns which subgraph works best. This reduced the cost by **1000×**.

The broader lesson: once a problem can be expressed as an optimization objective, a learning system can often find solutions that surprise human designers.

**Paper:** Pham, H., et al. (2018). "Efficient Neural Architecture Search via Parameter Sharing." *ICML 2018*.

</details>

<details>
<summary><strong>Geometric Deep Learning</strong></summary>

Most neural networks assume data lives on a regular grid (images) or a sequence (text). But much of the world's data has richer structure:

- **Graphs:** Social networks, molecules, knowledge bases
- **Manifolds:** 3D shapes, protein surfaces, climate data on a sphere
- **Groups:** Physical symmetries (rotation, translation, reflection)

Geometric Deep Learning provides a unified framework for building neural networks that respect these structures. The key insight (from Bronstein et al., 2021) is that most successful architectures — CNNs, RNNs, GNNs, Transformers — can be understood as special cases of a general principle: **equivariance to symmetry groups**.

- **CNNs** are equivariant to translations (shifting an image shifts the feature maps)
- **GNNs** are equivariant to node permutations (relabeling nodes doesn't change the output)
- **Transformers** are equivariant to input permutations (with positional encoding breaking this symmetry)

Applications: drug discovery (molecular graphs), protein design, weather prediction, particle physics.

**Paper:** Bronstein, M.M., et al. (2021). "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." *arXiv:2104.13478*.

</details>

<details>
<summary><strong>Honorable Mentions</strong></summary>

A few other directions worth knowing about:

**Diffusion Models** — Generate images (and other data) by learning to reverse a noise-adding process. Start with pure noise, iteratively denoise it into a coherent image. Powers DALL-E, Stable Diffusion, and Midjourney. The mathematical framework is based on stochastic differential equations and score matching.

**Mixture of Experts (MoE)** — Instead of one monolithic network, use many specialized "expert" sub-networks. A gating network routes each input to the most relevant experts. This allows models to be much larger (more total parameters) without proportionally increasing compute per input. Used in GPT-4 and Mixtral.

**State-Space Models (SSMs)** — An alternative to Transformers for sequence modeling. Models like Mamba process sequences in linear time (vs. Transformers' quadratic attention), making them appealing for very long sequences. Based on continuous-time dynamical systems discretized for digital computation.

**World Models** — Agents that learn an internal model of their environment, enabling planning and imagination. Rather than learning a direct mapping from states to actions, they learn to predict the consequences of actions, then plan by "imagining" future trajectories.

</details>

---

## References

### Papers

| Year | Authors | Title | Publication |
|------|---------|-------|-------------|
| 2016 | Silver et al. | [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://www.nature.com/articles/nature16961) | *Nature*, 529, 484–489 |
| 2017 | Silver et al. | [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) | *arXiv:1712.01815* |
| 2018 | Pham et al. | [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268) | *ICML 2018* |
| 2020 | Lewis et al. | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | *NeurIPS 2020* |
| 2021 | Hu et al. | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | *ICLR 2022* |
| 2021 | Jumper et al. | [Highly Accurate Protein Structure Prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) | *Nature*, 596, 583–589 |
| 2021 | Bronstein et al. | [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478) | *arXiv:2104.13478* |
| 2025 | Novikov et al. | [AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery](https://arxiv.org/abs/2506.13131) | *arXiv:2506.13131* |

### Datasets & Tools

- [MNIST Database](http://yann.lecun.com/exdb/mnist/) — Yann LeCun's original dataset page
- [Iris Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset) — scikit-learn documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
