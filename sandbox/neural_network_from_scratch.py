"""Neural network from scratch solving XOR — no libraries, just math.

This is the pedagogical centerpiece of Session 01. Every operation that
frameworks like PyTorch hide behind autograd is written out explicitly:
forward pass, loss computation, backpropagation, weight update.

Architecture: 2 inputs → 4 hidden neurons (sigmoid) → 1 output (sigmoid)
"""

from __future__ import annotations

import math
import random

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

console = Console()


def sigmoid(x: float) -> float:
    """Logistic sigmoid activation: squashes any value into (0, 1)."""
    # Clamp to avoid overflow in exp()
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(sigmoid_output: float) -> float:
    """Derivative of sigmoid given its *output* (not input): σ(x) · (1 − σ(x))."""
    return sigmoid_output * (1.0 - sigmoid_output)


class NeuralNetwork:
    """A minimal 2-layer network: input → hidden → output.

    All weights and biases are plain Python floats. Forward pass and
    backpropagation are spelled out step by step for clarity.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.5) -> None:
        self.learning_rate = learning_rate

        # Hidden layer weights: hidden_size × input_size
        self.weights_input_hidden: list[list[float]] = [
            [random.uniform(-1, 1) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.biases_hidden: list[float] = [random.uniform(-1, 1) for _ in range(hidden_size)]

        # Output layer weights: output_size × hidden_size
        self.weights_hidden_output: list[list[float]] = [
            [random.uniform(-1, 1) for _ in range(hidden_size)]
            for _ in range(output_size)
        ]
        self.biases_output: list[float] = [random.uniform(-1, 1) for _ in range(output_size)]

    def _forward(self, inputs: list[float]) -> tuple[list[float], list[float]]:
        """Compute hidden activations and output activations."""
        # Hidden layer
        hidden_activations: list[float] = []
        for neuron_index in range(len(self.weights_input_hidden)):
            weighted_sum = self.biases_hidden[neuron_index]
            for input_index in range(len(inputs)):
                weighted_sum += self.weights_input_hidden[neuron_index][input_index] * inputs[input_index]
            hidden_activations.append(sigmoid(weighted_sum))

        # Output layer
        output_activations: list[float] = []
        for neuron_index in range(len(self.weights_hidden_output)):
            weighted_sum = self.biases_output[neuron_index]
            for hidden_index in range(len(hidden_activations)):
                weighted_sum += self.weights_hidden_output[neuron_index][hidden_index] * hidden_activations[hidden_index]
            output_activations.append(sigmoid(weighted_sum))

        return hidden_activations, output_activations

    def predict(self, inputs: list[float]) -> list[float]:
        """Forward pass only — returns output activations."""
        _, outputs = self._forward(inputs)
        return outputs

    def train_step(self, inputs: list[float], targets: list[float]) -> float:
        """One forward + backward pass. Returns mean squared error."""
        hidden_activations, output_activations = self._forward(inputs)

        # --- Output layer error ---
        output_errors: list[float] = []
        for index in range(len(output_activations)):
            error = targets[index] - output_activations[index]
            output_errors.append(error)

        # Mean squared error for reporting
        mse = sum(e ** 2 for e in output_errors) / len(output_errors)

        # --- Output layer gradients ---
        output_deltas: list[float] = []
        for index in range(len(output_activations)):
            delta = output_errors[index] * sigmoid_derivative(output_activations[index])
            output_deltas.append(delta)

        # --- Hidden layer error (backpropagated from output) ---
        hidden_errors: list[float] = [0.0] * len(hidden_activations)
        for output_index in range(len(output_deltas)):
            for hidden_index in range(len(hidden_activations)):
                hidden_errors[hidden_index] += output_deltas[output_index] * self.weights_hidden_output[output_index][hidden_index]

        hidden_deltas: list[float] = []
        for index in range(len(hidden_activations)):
            delta = hidden_errors[index] * sigmoid_derivative(hidden_activations[index])
            hidden_deltas.append(delta)

        # --- Update output weights and biases ---
        for output_index in range(len(output_deltas)):
            for hidden_index in range(len(hidden_activations)):
                self.weights_hidden_output[output_index][hidden_index] += (
                    self.learning_rate * output_deltas[output_index] * hidden_activations[hidden_index]
                )
            self.biases_output[output_index] += self.learning_rate * output_deltas[output_index]

        # --- Update hidden weights and biases ---
        for hidden_index in range(len(hidden_deltas)):
            for input_index in range(len(inputs)):
                self.weights_input_hidden[hidden_index][input_index] += (
                    self.learning_rate * hidden_deltas[hidden_index] * inputs[input_index]
                )
            self.biases_hidden[hidden_index] += self.learning_rate * hidden_deltas[hidden_index]

        return mse


def run() -> None:
    """Train a from-scratch neural network on XOR and display results."""
    console.print(
        Panel(
            "[bold]Neural Network From Scratch[/bold]\n"
            "2 inputs → 4 hidden (sigmoid) → 1 output (sigmoid)\n"
            "No libraries — just Python, math, and backpropagation.",
            border_style="green",
        )
    )

    xor_data: list[tuple[list[float], list[float]]] = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]

    random.seed(42)
    network = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=2.0)

    epochs = 10_000
    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Training...", total=epochs)
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, targets in xor_data:
                total_loss += network.train_step(inputs, targets)
            average_loss = total_loss / len(xor_data)

            if epoch % 1000 == 0:
                progress.update(task, advance=1000, description=f"[cyan]Epoch {epoch:>5}  Loss: {average_loss:.6f}")
        progress.update(task, completed=epochs)

    # Show final predictions
    console.print()
    table = Table(title="XOR Predictions (from scratch)", show_header=True, header_style="bold green")
    table.add_column("Input A", justify="center")
    table.add_column("Input B", justify="center")
    table.add_column("Expected", justify="center")
    table.add_column("Raw Output", justify="center")
    table.add_column("Rounded", justify="center")
    table.add_column("", justify="center")

    for inputs, targets in xor_data:
        output = network.predict(inputs)[0]
        rounded = round(output)
        expected = int(targets[0])
        status = "[green]✓[/green]" if rounded == expected else "[red]✗[/red]"
        table.add_row(
            str(int(inputs[0])),
            str(int(inputs[1])),
            str(expected),
            f"{output:.4f}",
            str(rounded),
            status,
        )

    console.print(table)
    console.print(
        "\n[dim]Every operation here — forward pass, loss, gradients, weight updates —"
        " is exactly what PyTorch's autograd does for you automatically.[/dim]\n"
    )
