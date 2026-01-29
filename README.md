# Playing With Sand In Black Boxes

A multi-session workshop teaching AI/ML from first principles — from hand-built perceptrons to agentic systems.

Two tracks: a 3-session technical route and a 1-2 session non-technical user route.

## Sessions

| Session | Topic | Material | Demos |
|---------|-------|----------|-------|
| **01** | [Foundations](session_01_foundations.md) | Perceptrons, backprop from scratch, PyTorch intro | `session01-perceptron`, `session01-nn-scratch`, `session01-xor-pytorch` |
| **02** | [Modern Approaches](session_02_modern_approaches.md) | MNIST CNN, custom models, fine-tuning, RAG, research frontiers | `session02-mnist`, `session02-custom` |
| **03** | [Agentic Systems](session_03_agentic_systems.md) | ReAct agents, tool use, agentic SaaS landscape | `session03-agent` |

## Setup

Requires Python 3.12 and [UV](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Run the interactive menu
uv run sand

# Or run a specific demo directly
uv run sand session01-perceptron
```

## CLI Commands

```
uv run sand                       Interactive menu
uv run sand session01-perceptron  Single-layer perceptron (AND, OR, XOR)
uv run sand session01-nn-scratch  Neural network from scratch (XOR)
uv run sand session01-xor-pytorch PyTorch XOR solution
uv run sand session02-mnist       CNN on MNIST digits
uv run sand session02-custom      Feedforward on Iris dataset
uv run sand session03-agent       Mock ReAct agent loop
```

## Dependencies

- **rich** — terminal UI (tables, panels, progress bars)
- **torch** — PyTorch deep learning framework
- **torchvision** — datasets and transforms (MNIST)
- **scikit-learn** — datasets and preprocessing (Iris)
