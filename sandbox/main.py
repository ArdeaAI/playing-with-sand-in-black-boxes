"""Interactive CLI menu for Playing With Sand In Black Boxes workshop demos."""

from __future__ import annotations

import importlib
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

DEMOS: dict[str, tuple[str, str]] = {
    "session01-perceptron": ("perceptron", "Single-layer perceptron — AND, OR, and why XOR fails"),
    "session01-nn-scratch": ("neural_network_from_scratch", "Neural network from scratch solving XOR"),
    "session01-xor-pytorch": ("xor_pytorch", "PyTorch XOR solution"),
    "session02-mnist": ("mnist_model", "CNN on MNIST handwritten digits"),
    "session02-custom": ("custom_model", "Feedforward network on Iris dataset"),
    "session03-agent": ("simple_agent", "Mock ReAct agent loop"),
}


def _run_demo(command: str) -> None:
    """Lazy-import and run a demo module by command name."""
    if command not in DEMOS:
        console.print(f"[red]Unknown command:[/red] {command}")
        console.print("Run [bold]sand[/bold] with no arguments to see available demos.")
        sys.exit(1)

    module_name, description = DEMOS[command]
    console.print(f"\n[bold cyan]▶ {description}[/bold cyan]\n")
    module = importlib.import_module(f"sandbox.{module_name}")
    module.run()


def _interactive_menu() -> None:
    """Display an interactive menu and run the selected demo."""
    console.print(
        Panel(
            "[bold]Playing With Sand In Black Boxes[/bold]\n"
            "A workshop from hand-built perceptrons to agentic systems",
            border_style="bright_blue",
        )
    )

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Command", style="cyan")
    table.add_column("Description")

    commands = list(DEMOS.keys())
    for index, command in enumerate(commands, 1):
        _, description = DEMOS[command]
        table.add_row(str(index), command, description)

    console.print(table)
    console.print()

    choice = console.input("[bold]Enter number or command name (q to quit): [/bold]").strip()

    if choice.lower() in ("q", "quit", "exit"):
        return

    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(commands):
            _run_demo(commands[index])
        else:
            console.print(f"[red]Invalid choice:[/red] {choice}")
    else:
        _run_demo(choice)


def main() -> None:
    """Entry point: dispatch by sys.argv or show interactive menu."""
    if len(sys.argv) > 1:
        _run_demo(sys.argv[1])
    else:
        _interactive_menu()


if __name__ == "__main__":
    main()
