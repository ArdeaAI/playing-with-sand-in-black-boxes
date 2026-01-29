"""PyTorch XOR solution — contrast with the from-scratch version.

Same problem (XOR), same idea (2-layer network with nonlinear activation),
but PyTorch handles the gradient math via autograd. This demo shows how
frameworks compress the manual work into a few declarative lines.

Architecture: Linear(2,4) → ReLU → Linear(4,1) → Sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

console = Console()


class XORModel(nn.Module):
    """Minimal feedforward network for XOR."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


def run() -> None:
    """Train the PyTorch XOR model and display results."""
    console.print(
        Panel(
            "[bold]XOR with PyTorch[/bold]\n"
            "Linear(2,4) → ReLU → Linear(4,1) → Sigmoid\n"
            "Same problem as the from-scratch version, but autograd handles backprop.",
            border_style="blue",
        )
    )

    # Data
    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    # Model, loss, optimizer
    torch.manual_seed(42)
    model = XORModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # Training loop
    epochs = 5000
    with Progress(console=console) as progress:
        task = progress.add_task("[blue]Training...", total=epochs)
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0:
                progress.update(
                    task,
                    advance=500,
                    description=f"[blue]Epoch {epoch:>5}  Loss: {loss.item():.6f}",
                )
        progress.update(task, completed=epochs)

    # Results — switch to inference mode (no gradient tracking)
    console.print()
    model.train(False)
    with torch.no_grad():
        final_predictions = model(inputs)

    table = Table(title="XOR Predictions (PyTorch)", show_header=True, header_style="bold blue")
    table.add_column("Input A", justify="center")
    table.add_column("Input B", justify="center")
    table.add_column("Expected", justify="center")
    table.add_column("Raw Output", justify="center")
    table.add_column("Rounded", justify="center")
    table.add_column("", justify="center")

    for index in range(len(inputs)):
        raw = final_predictions[index].item()
        rounded = round(raw)
        expected = int(targets[index].item())
        status = "[green]✓[/green]" if rounded == expected else "[red]✗[/red]"
        table.add_row(
            str(int(inputs[index][0].item())),
            str(int(inputs[index][1].item())),
            str(expected),
            f"{raw:.4f}",
            str(rounded),
            status,
        )

    console.print(table)

    console.print(
        "\n[dim]Compare this with the from-scratch version: same architecture, same result,"
        " but PyTorch computed all the gradients automatically.[/dim]\n"
    )
