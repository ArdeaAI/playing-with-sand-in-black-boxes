# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-session workshop teaching AI/ML from first principles — from hand-built perceptrons to agentic systems. Two tracks: a 3-session technical route and a 1-2 session non-technical user route. The technical route progresses through:

1. **Session 01 (Foundations)**: Historical context, raw-Python perceptron solving XOR, intro to PyTorch
2. **Session 02 (Modern Approaches)**: MNIST, custom models, frontier LLM techniques (fine-tuning, RAG), research directions (Alpha Series, ENAS, Geometric Models)
3. **Session 03 (Agentic Systems)**: What agents are, building one, surveying SaaS agentic tools

## Build & Run

```bash
# Environment setup (UV, Python 3.12)
uv sync

# Interactive menu
uv run sand

# Run a specific demo
uv run sand session01-perceptron
uv run sand session01-nn-scratch
uv run sand session01-xor-pytorch
uv run sand session02-mnist
uv run sand session02-custom
uv run sand session03-agent
```

## Architecture

```
playing-with-sand-in-black-boxes/
├── pyproject.toml                         # UV project config, dependencies, entry point
├── main.py                                # Root entry point (delegates to sandbox.main)
├── session_01_foundations.md              # Session 01 workshop content (collapsible sections)
├── session_02_modern_approaches.md       # Session 02 workshop content
├── session_03_agentic_systems.md         # Session 03 workshop content
├── sandbox/
│   ├── main.py                           # Rich CLI menu + sys.argv command dispatch
│   ├── perceptron.py                     # Single-layer perceptron demo (stdlib + rich)
│   ├── neural_network_from_scratch.py    # Pure Python NN solving XOR (stdlib + rich)
│   ├── xor_pytorch.py                    # PyTorch XOR solution (torch + rich)
│   ├── mnist_model.py                    # CNN on MNIST (torch + torchvision + rich)
│   ├── custom_model.py                   # Feedforward on Iris (torch + sklearn + rich)
│   └── simple_agent.py                   # Mock ReAct agent loop (stdlib + rich)
└── ai_ref/                               # Workshop planning/reference materials (gitignored)
```

### Key Design Decisions

- **No `__init__.py`**: Modern Python, hatchling handles packaging
- **Flat sandbox structure**: 6 modules, no subdirectories needed
- **Lazy imports in CLI**: `importlib.import_module` so PyTorch doesn't load for the menu
- **Each module exposes `run()`**: Standard interface called by the CLI dispatcher
- **Mock agent (no API keys)**: `simple_agent.py` uses pattern matching, runs fully offline

## Conventions

- Python `>=3.12` strictly (see `.python-version` and `pyproject.toml`)
- Workshop code should be runnable standalone per session — each session's material works independently
- Code prioritizes clarity and pedagogical value — explicit is better than clever
- Rich library for all terminal output (tables, panels, progress bars)
- The `ai_ref/` directory contains WIP outlines and planning docs; it is gitignored

## Tooling

- **Linting/Formatting**: Ruff (line length 120)
- **Type Checking**: MyPy strict
- **Testing**: pytest
- **Package Manager**: UV (never pip)
