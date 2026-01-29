# Session 01: Foundations

From a single neuron to a network that learns — the ideas behind modern AI, built from scratch in Python.

**Demos:**
- `uv run sand session01-perceptron` — single-layer perceptron on AND, OR, XOR
- `uv run sand session01-nn-scratch` — neural network from scratch solving XOR
- `uv run sand session01-xor-pytorch` — PyTorch XOR solution

---

## Overview of the Workshop Series

This workshop teaches AI/ML from first principles. We start with the simplest possible neural network (one neuron, a step function) and build up to agentic systems that reason and act autonomously.

| Session | Focus | Key Idea |
|---------|-------|----------|
| **01 — Foundations** | Perceptrons, backprop, PyTorch intro | How machines learn from data |
| **02 — Modern Approaches** | CNNs, custom models, frontier techniques | What modern AI can do |
| **03 — Agentic Systems** | ReAct agents, tool use, SaaS landscape | AI that takes action |

---

## The Perceptron

A perceptron is the simplest neural network: one neuron with weighted inputs, a bias, and a step-function activation.

```
inputs × weights + bias → step function → output (0 or 1)
```

<details>
<summary><strong>How a Perceptron Works (4-neuron example)</strong></summary>

Imagine four inputs, each with a weight. The perceptron computes a weighted sum, adds a bias, and passes the result through a step function:

1. **Weighted sum:** `z = w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + b`
2. **Activation:** `output = 1 if z > 0, else 0`

During training, the perceptron adjusts its weights using the **perceptron learning rule**:

```
error = expected - predicted
wᵢ = wᵢ + learning_rate × error × xᵢ
```

If the prediction is correct, weights don't change. If wrong, they shift toward the correct answer. After enough iterations over the training data, the perceptron converges — *if* the data is linearly separable.

</details>

<details>
<summary><strong>The XOR Problem — Why One Neuron Isn't Enough</strong></summary>

AND and OR are **linearly separable** — you can draw a single straight line separating the 0s from the 1s on a 2D plot. A perceptron can learn this line.

XOR is **not** linearly separable. The positive examples (0,1) and (1,0) are on opposite corners; no single line divides them from (0,0) and (1,1). A perceptron will cycle forever without converging.

This is what Minsky & Papert proved formally in 1969, and it's the reason multi-layer networks exist. Adding a hidden layer gives the network enough representational power to learn XOR — which is exactly what we do in the next demo.

**Geometric intuition:** Each neuron in a hidden layer draws one line. Multiple lines together can carve out non-convex regions, solving XOR and far more complex problems.

</details>

---

## Brief History of Neural Networks

<details>
<summary><strong>McCulloch & Pitts (1943) — The First Artificial Neuron</strong></summary>

Warren McCulloch (neuroscientist) and Walter Pitts (logician) proposed the first mathematical model of a neuron. Their model was binary: a neuron either fires or doesn't, based on whether the weighted sum of its inputs exceeds a threshold.

They proved that networks of these simple units could compute any logical function — making them equivalent to a Turing machine. This paper launched the entire field of artificial neural networks, though the neurons were hand-wired, not learned.

**Paper:** McCulloch, W.S. & Pitts, W. (1943). "A Logical Calculus of the Ideas Immanent in Nervous Activity." *Bulletin of Mathematical Biophysics*, 5, 115–133.

</details>

<details>
<summary><strong>Frank Rosenblatt (1958) — The Perceptron</strong></summary>

Rosenblatt, a psychologist at Cornell, built on McCulloch-Pitts by adding a **learning rule**. His perceptron could adjust its weights from data — the first machine that genuinely *learned*.

He implemented it on the Mark I Perceptron, a hardware device with 400 photocells (a 20×20 pixel camera), potentiometers for weights, and electric motors to adjust them during training. The media called it a machine that could "learn to recognize things."

The perceptron convergence theorem guarantees that if the data is linearly separable, the learning rule will find a solution in finite steps.

**Paper:** Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain." *Psychological Review*, 65(6), 386–408.

</details>

<details>
<summary><strong>Widrow & Hoff (1960) — ADALINE and the LMS Rule</strong></summary>

Bernard Widrow and Marcian Hoff at Stanford introduced ADALINE (Adaptive Linear Neuron) and the Least Mean Squares (LMS) learning rule — also known as the Widrow-Hoff rule or the delta rule.

Unlike Rosenblatt's perceptron (which adjusts weights based on the binary output), LMS uses the *continuous* error signal before the threshold function. This makes it a gradient-based method — a direct ancestor of modern backpropagation.

ADALINE found practical use in adaptive filters (noise cancellation, echo suppression in phone lines) and is still used in signal processing.

**Paper:** Widrow, B. & Hoff, M.E. (1960). "Adaptive Switching Circuits." *IRE WESCON Convention Record*, 4, 96–104.

</details>

<details>
<summary><strong>Minsky & Papert (1969) — Perceptrons and the First AI Winter</strong></summary>

Marvin Minsky and Seymour Papert published *Perceptrons: An Introduction to Computational Geometry*, which rigorously proved that single-layer perceptrons cannot solve problems like XOR, parity, or connectedness.

The book's impact went beyond its mathematical content. Funding agencies interpreted it as evidence that neural networks were a dead end, and research funding dried up for over a decade. This period is known as the **first AI winter** (roughly 1969–1980).

The irony: multi-layer networks *could* solve these problems, but no one had a practical algorithm to train them yet.

**Book:** Minsky, M. & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry.* MIT Press.

</details>

<details>
<summary><strong>Symbolic AI Era (1970s–80s) — Expert Systems</strong></summary>

With neural networks in decline, AI research shifted to **symbolic methods**: hand-coded rules, logic programming, and expert systems. These systems encoded human knowledge as if-then rules.

Notable examples:
- **MYCIN** (1976) — diagnosed bacterial infections using ~600 rules
- **R1/XCON** (1980) — configured DEC VAX computer orders, saving DEC ~$40M/year
- **Prolog** — logic programming language that became the lingua franca of AI

Expert systems worked well in narrow domains but were brittle: they couldn't generalize, couldn't learn from data, and required enormous manual effort to build and maintain. By the late 1980s, the limits of symbolic AI triggered a **second AI winter**.

</details>

<details>
<summary><strong>Rumelhart, Hinton & Williams (1986) — Backpropagation</strong></summary>

The paper "Learning Representations by Back-Propagating Errors" showed that multi-layer networks could be trained by propagating error gradients backward through the network — **backpropagation**.

The key insight: using the chain rule of calculus, you can compute how much each weight in any layer contributed to the final error, then adjust it proportionally. This is exactly what our from-scratch demo implements manually.

Backpropagation didn't appear from nowhere — Paul Werbos described it in his 1974 PhD thesis, and similar ideas existed in control theory. But Rumelhart, Hinton, and Williams demonstrated it convincingly on practical problems and made it accessible to the research community.

This paper revived neural network research after 15 years of dormancy.

**Paper:** Rumelhart, D.E., Hinton, G.E. & Williams, R.J. (1986). "Learning Representations by Back-Propagating Errors." *Nature*, 323, 533–536.

</details>

<details>
<summary><strong>Vaswani et al. (2017) — "Attention Is All You Need"</strong></summary>

The Transformer architecture replaced recurrence (RNNs, LSTMs) with **self-attention** — a mechanism that lets every token in a sequence attend to every other token in parallel.

Why this matters:
- **Parallelism:** Unlike RNNs, which process tokens sequentially, Transformers process entire sequences at once on GPUs
- **Long-range dependencies:** Attention connects distant tokens directly, without information having to flow through intermediate steps
- **Scalability:** The architecture scales with data and compute, enabling modern LLMs with billions of parameters

Every major LLM today (GPT, Claude, Gemini, LLaMA) is built on the Transformer.

**Paper:** Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*.

</details>

<details>
<summary><strong>Stanley & Miikkulainen (2002) — NEAT</strong></summary>

NeuroEvolution of Augmenting Topologies (NEAT) takes a completely different approach: instead of training a fixed architecture with gradient descent, it **evolves** both the topology and weights of neural networks using genetic algorithms.

Key ideas:
- **Start minimal:** Begin with the simplest possible network and add complexity only when needed
- **Speciation:** Protect novel structures from being eliminated before they've had time to optimize
- **Historical markings:** Track which genes correspond across different network topologies to enable meaningful crossover

NEAT won the Outstanding Paper of the Decade award (2002–2012) from the International Society for Artificial Life. It's a reminder that gradient descent isn't the only way to train neural networks.

**Paper:** Stanley, K.O. & Miikkulainen, R. (2002). "Evolving Neural Networks through Augmenting Topologies." *Evolutionary Computation*, 10(2), 99–127.

</details>

---

## Neural Network from Scratch — Code Walkthrough

The `neural_network_from_scratch.py` demo implements every operation that PyTorch hides:

1. **Forward pass:** Input → hidden layer (weighted sum + sigmoid) → output layer (weighted sum + sigmoid)
2. **Loss computation:** Mean squared error between predicted and expected outputs
3. **Backpropagation:** Compute gradients layer by layer using the chain rule
4. **Weight update:** Adjust each weight proportionally to its gradient × learning rate

<details>
<summary><strong>Architecture Details</strong></summary>

```
Input Layer (2 neurons)
    ↓ weights_input_hidden (4×2 matrix)
Hidden Layer (4 neurons, sigmoid activation)
    ↓ weights_hidden_output (1×4 matrix)
Output Layer (1 neuron, sigmoid activation)
```

**Sigmoid activation:** `σ(x) = 1 / (1 + e^(-x))`

Sigmoid squashes any value into (0, 1), making it interpretable as a probability. Its derivative `σ(x) × (1 - σ(x))` is needed for backpropagation and can be computed from the *output* alone (no need to store the input).

**Why 4 hidden neurons?** XOR needs at least 2 hidden neurons (to draw 2 decision boundaries). We use 4 for faster convergence — the extra capacity makes it easier for random initialization to land in a good spot.

</details>

<details>
<summary><strong>The Backpropagation Algorithm, Step by Step</strong></summary>

1. **Output error:** `error = target - output`
2. **Output delta:** `delta = error × sigmoid_derivative(output)`
3. **Hidden error:** Propagate output deltas backward through weights
4. **Hidden delta:** `delta = hidden_error × sigmoid_derivative(hidden_output)`
5. **Update output weights:** `w += learning_rate × output_delta × hidden_activation`
6. **Update hidden weights:** `w += learning_rate × hidden_delta × input`

This is the chain rule applied recursively. Each layer's gradient depends on the layer above it — hence "back-propagation."

</details>

---

## Solving XOR with PyTorch — Code Walkthrough

The `xor_pytorch.py` demo solves the same problem but replaces all manual math with PyTorch's autograd.

<details>
<summary><strong>What PyTorch Does For You</strong></summary>

| From-Scratch Step | PyTorch Equivalent |
|---|---|
| Manual weighted sum + sigmoid | `nn.Linear` + `torch.relu` / `torch.sigmoid` |
| Manual MSE calculation | `nn.BCELoss()` |
| Manual gradient computation | `loss.backward()` (autograd) |
| Manual weight update | `optimizer.step()` (Adam) |

PyTorch builds a **computation graph** during the forward pass. When you call `loss.backward()`, it walks this graph in reverse, computing gradients for every parameter automatically.

The model is identical in structure: `Linear(2,4) → ReLU → Linear(4,1) → Sigmoid`. The difference is that we define *what* to compute, and PyTorch figures out *how* to train it.

</details>

<details>
<summary><strong>Why ReLU Instead of Sigmoid?</strong></summary>

The from-scratch version uses sigmoid everywhere. The PyTorch version uses **ReLU** (Rectified Linear Unit) in the hidden layer: `ReLU(x) = max(0, x)`.

ReLU is preferred in modern networks because:
- **No vanishing gradient:** Sigmoid's gradient approaches zero for large/small inputs, making deep networks hard to train. ReLU's gradient is either 0 or 1.
- **Faster computation:** A comparison vs. an exponential.
- **Sparsity:** ReLU outputs zero for negative inputs, creating sparse activations.

We keep sigmoid on the output because we need a probability in (0, 1) for binary classification.

</details>

---

## References

### Papers

| Year | Authors | Title | Publication |
|------|---------|-------|-------------|
| 1943 | McCulloch & Pitts | [A Logical Calculus of the Ideas Immanent in Nervous Activity](https://link.springer.com/article/10.1007/BF02478259) | *Bulletin of Mathematical Biophysics*, 5, 115–133 |
| 1958 | Rosenblatt | [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://psycnet.apa.org/record/1959-09865-001) | *Psychological Review*, 65(6), 386–408 |
| 1960 | Widrow & Hoff | [Adaptive Switching Circuits](https://www-isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf) | *IRE WESCON Convention Record*, 4, 96–104 |
| 1969 | Minsky & Papert | [Perceptrons: An Introduction to Computational Geometry](https://mitpress.mit.edu/9780262630221/perceptrons/) | MIT Press |
| 1986 | Rumelhart, Hinton & Williams | [Learning Representations by Back-Propagating Errors](https://www.nature.com/articles/323533a0) | *Nature*, 323, 533–536 |
| 2002 | Stanley & Miikkulainen | [Evolving Neural Networks through Augmenting Topologies](https://direct.mit.edu/evco/article/10/2/99/1123/) | *Evolutionary Computation*, 10(2), 99–127 |
| 2017 | Vaswani et al. | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | *NeurIPS 2017* |

### Tools & Frameworks

- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
