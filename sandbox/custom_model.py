"""Feedforward network on the Iris dataset — applying PyTorch to tabular data.

Iris is a classic dataset: 150 samples, 4 features (sepal/petal length and
width), 3 species classes. It demonstrates that neural networks aren't just
for images — the same train/predict pattern applies to any numeric data.

Architecture: Linear(4,16) → ReLU → Linear(16,3) → CrossEntropyLoss
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

console = Console()


class IrisNet(nn.Module):
    """Simple feedforward classifier for 4 features → 3 classes."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 16)
        self.output = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x


def run() -> None:
    """Train a feedforward network on Iris and display results."""
    console.print(
        Panel(
            "[bold]Custom Model — Iris Dataset[/bold]\n"
            "Linear(4,16) → ReLU → Linear(16,3)\n"
            "150 flowers, 4 measurements, 3 species.",
            border_style="yellow",
        )
    )

    # Load and split data
    iris = load_iris()
    feature_names: list[str] = iris.feature_names  # type: ignore[attr-defined]
    target_names: list[str] = list(iris.target_names)  # type: ignore[attr-defined]

    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target  # type: ignore[attr-defined]
    )

    # Standardize features (important for neural networks)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Convert to tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    console.print(f"[dim]Training samples: {len(x_train_tensor)}  Test samples: {len(x_test_tensor)}[/dim]")
    console.print(f"[dim]Features: {', '.join(feature_names)}[/dim]")
    console.print(f"[dim]Classes: {', '.join(target_names)}[/dim]\n")

    # Model setup
    torch.manual_seed(42)
    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    epochs = 200
    with Progress(console=console) as progress:
        task = progress.add_task("[yellow]Training...", total=epochs)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                progress.update(
                    task,
                    advance=20,
                    description=f"[yellow]Epoch {epoch:>4}  Loss: {loss.item():.4f}",
                )
        progress.update(task, completed=epochs)

    # Testing
    model.train(False)
    with torch.no_grad():
        test_outputs = model(x_test_tensor)
        _, predicted = torch.max(test_outputs, 1)

    correct = (predicted == y_test_tensor).sum().item()
    total = len(y_test_tensor)
    accuracy = 100.0 * correct / total

    console.print(f"\n[bold green]Test accuracy: {accuracy:.1f}%[/bold green]  ({correct}/{total})\n")

    # Per-class results
    table = Table(title="Accuracy by Species", show_header=True, header_style="bold yellow")
    table.add_column("Species", justify="left")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right")

    for class_index, species_name in enumerate(target_names):
        class_mask = y_test_tensor == class_index
        class_total = class_mask.sum().item()
        class_correct = ((predicted == y_test_tensor) & class_mask).sum().item()
        class_accuracy = 100.0 * class_correct / class_total if class_total > 0 else 0.0
        table.add_row(
            species_name,
            str(class_correct),
            str(class_total),
            f"{class_accuracy:.1f}%",
        )

    console.print(table)
    console.print(
        "\n[dim]Neural networks work on any numeric data, not just images."
        " The same forward-pass → loss → backprop → update pattern applies everywhere.[/dim]\n"
    )
