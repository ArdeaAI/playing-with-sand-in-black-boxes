"""Single-layer perceptron demonstrating AND, OR, and the XOR problem.

A perceptron is the simplest possible neural network — one neuron with a
step-function activation. It can learn any *linearly separable* function,
which is why AND and OR succeed but XOR fails. That failure motivated the
development of multi-layer networks.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class Perceptron:
    """Single-layer perceptron with step-function activation."""

    def __init__(self, input_size: int, learning_rate: float = 0.1) -> None:
        self.weights: list[float] = [0.0] * input_size
        self.bias: float = 0.0
        self.learning_rate = learning_rate

    def predict(self, inputs: list[float]) -> int:
        """Weighted sum → step function (threshold at 0)."""
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return 1 if weighted_sum > 0 else 0

    def train(self, training_data: list[tuple[list[float], int]], epochs: int = 100) -> None:
        """Perceptron learning rule: adjust weights by error × input."""
        for _ in range(epochs):
            for inputs, expected in training_data:
                prediction = self.predict(inputs)
                error = expected - prediction
                for index in range(len(self.weights)):
                    self.weights[index] += self.learning_rate * error * inputs[index]
                self.bias += self.learning_rate * error


def _demo_gate(gate_name: str, training_data: list[tuple[list[float], int]]) -> bool:
    """Train a perceptron on a logic gate and display results. Returns True if all predictions match."""
    perceptron = Perceptron(input_size=2)
    perceptron.train(training_data, epochs=100)

    table = Table(title=f"{gate_name} Gate", show_header=True, header_style="bold green")
    table.add_column("Input A", justify="center")
    table.add_column("Input B", justify="center")
    table.add_column("Expected", justify="center")
    table.add_column("Predicted", justify="center")
    table.add_column("", justify="center")

    all_correct = True
    for inputs, expected in training_data:
        predicted = perceptron.predict(inputs)
        correct = predicted == expected
        if not correct:
            all_correct = False
        status = "[green]✓[/green]" if correct else "[red]✗[/red]"
        table.add_row(
            str(int(inputs[0])),
            str(int(inputs[1])),
            str(expected),
            str(predicted),
            status,
        )

    console.print(table)
    console.print(
        f"  Weights: {perceptron.weights}  Bias: {perceptron.bias:.2f}\n"
    )
    return all_correct


def run() -> None:
    """Run all three gate demos."""
    console.print(
        Panel(
            "[bold]The Perceptron[/bold]\n"
            "One neuron, a step function, and the limits of linear separability.",
            border_style="yellow",
        )
    )

    # AND gate — linearly separable, should succeed
    and_data: list[tuple[list[float], int]] = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1),
    ]
    _demo_gate("AND", and_data)

    # OR gate — linearly separable, should succeed
    or_data: list[tuple[list[float], int]] = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1),
    ]
    _demo_gate("OR", or_data)

    # XOR gate — NOT linearly separable, perceptron will fail
    xor_data: list[tuple[list[float], int]] = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]
    xor_passed = _demo_gate("XOR", xor_data)

    if not xor_passed:
        console.print(
            Panel(
                "[bold red]XOR failed![/bold red] A single perceptron can only learn "
                "[italic]linearly separable[/italic] functions.\n\n"
                "XOR requires a decision boundary that no single straight line can draw. "
                "This is the problem that motivated multi-layer neural networks — which "
                "we build from scratch in the next demo.",
                title="Why XOR Fails",
                border_style="red",
            )
        )
