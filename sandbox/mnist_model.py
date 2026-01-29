"""CNN on MNIST handwritten digits — the "hello world" of deep learning.

MNIST is a dataset of 70,000 grayscale 28×28 images of handwritten digits
(0-9). A convolutional neural network learns spatial features (edges, curves,
loops) that generalize across handwriting styles.

Architecture: Conv2d → ReLU → MaxPool → Conv2d → ReLU → MaxPool → FC → FC → output
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

console = Console()


class MNISTConvNet(nn.Module):
    """Small CNN for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
        # First conv block: 1 channel in → 16 filters, 3×3 kernel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Second conv block: 16 → 32 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # After two 2×2 max-pools: 28→14→7, so flattened = 32 * 7 * 7
        self.fully_connected_1 = nn.Linear(32 * 7 * 7, 128)
        self.fully_connected_2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1: conv → relu → pool
        x = functional.max_pool2d(functional.relu(self.conv1(x)), 2)
        # Conv block 2: conv → relu → pool
        x = functional.max_pool2d(functional.relu(self.conv2(x)), 2)
        # Flatten spatial dims
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = functional.relu(self.fully_connected_1(x))
        x = self.fully_connected_2(x)
        return x


def run() -> None:
    """Train a CNN on MNIST and report accuracy."""
    console.print(
        Panel(
            "[bold]MNIST Handwritten Digit Recognition[/bold]\n"
            "Conv2d(1→16) → Conv2d(16→32) → FC(1568→128) → FC(128→10)\n"
            "70,000 images of handwritten digits, 28×28 pixels each.",
            border_style="magenta",
        )
    )

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    console.print(f"[dim]Using device: {device}[/dim]\n")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
    ])

    console.print("[dim]Downloading MNIST dataset (if needed)...[/dim]")
    training_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    training_loader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model setup
    torch.manual_seed(42)
    model = MNISTConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_parameters = sum(p.numel() for p in model.parameters())
    console.print(f"[dim]Model parameters: {total_parameters:,}[/dim]\n")

    # Training
    epochs = 3
    with Progress(console=console) as progress:
        for epoch in range(1, epochs + 1):
            model.train()
            task = progress.add_task(f"[magenta]Epoch {epoch}/{epochs}", total=len(training_loader))
            running_loss = 0.0

            for batch_index, (images, labels) in enumerate(training_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress.update(task, advance=1)

            average_loss = running_loss / len(training_loader)
            progress.update(task, description=f"[magenta]Epoch {epoch}/{epochs}  Loss: {average_loss:.4f}")

    # Testing
    model.train(False)
    correct = 0
    total = 0
    per_class_correct: dict[int, int] = {i: 0 for i in range(10)}
    per_class_total: dict[int, int] = {i: 0 for i in range(10)}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                digit = label.item()
                per_class_total[digit] += 1
                if prediction.item() == digit:
                    per_class_correct[digit] += 1

    overall_accuracy = 100.0 * correct / total
    console.print(f"\n[bold green]Test accuracy: {overall_accuracy:.2f}%[/bold green]  ({correct}/{total})\n")

    # Per-digit breakdown
    table = Table(title="Accuracy by Digit", show_header=True, header_style="bold magenta")
    table.add_column("Digit", justify="center")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right")

    for digit in range(10):
        digit_accuracy = 100.0 * per_class_correct[digit] / per_class_total[digit]
        table.add_row(
            str(digit),
            str(per_class_correct[digit]),
            str(per_class_total[digit]),
            f"{digit_accuracy:.1f}%",
        )

    console.print(table)
    console.print(
        "\n[dim]Three epochs on a small CNN already achieves ~98-99% accuracy."
        " The convolutional layers learn edge and shape detectors automatically.[/dim]\n"
    )
