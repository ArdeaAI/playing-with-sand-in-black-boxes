"""Mock ReAct agent loop — teaching the agentic pattern without API keys.

A ReAct agent follows the cycle: Reason → Act → Observe. The "LLM" here is
a simple pattern-matcher so the demo runs fully offline. The architecture is
real — swap in an actual LLM client and these tools would work as-is.

Pattern: The agent receives a question, reasons about which tool to use,
calls the tool, observes the result, and repeats until it has an answer.
"""

from __future__ import annotations

import datetime
import math
import re
from dataclasses import dataclass
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class Tool:
    """A tool the agent can invoke."""

    name: str
    description: str
    function: Callable[[str], str]


def _parse_arithmetic(expression: str) -> float:
    """Parse and compute a basic arithmetic expression using recursive descent.

    Supports +, -, *, /, ** (exponentiation), parentheses, and unary minus.
    No use of Python's built-in code-execution facilities.
    """
    tokens: list[str] = []
    current = ""
    previous_char = ""
    for char in expression:
        if char in "+-*/()":
            if char == "*" and previous_char == "*":
                # Handle ** (exponentiation) — merge with previous token
                tokens[-1] = "**"
                previous_char = ""
                current = ""
                continue
            if current.strip():
                tokens.append(current.strip())
                current = ""
            tokens.append(char)
        else:
            current += char
        previous_char = char
    if current.strip():
        tokens.append(current.strip())

    position = [0]

    def peek() -> str | None:
        return tokens[position[0]] if position[0] < len(tokens) else None

    def consume() -> str:
        token = tokens[position[0]]
        position[0] += 1
        return token

    def parse_expression() -> float:
        result = parse_term()
        while peek() in ("+", "-"):
            operator = consume()
            right = parse_term()
            result = result + right if operator == "+" else result - right
        return result

    def parse_term() -> float:
        result = parse_power()
        while peek() in ("*", "/"):
            operator = consume()
            right = parse_power()
            result = result * right if operator == "*" else result / right
        return result

    def parse_power() -> float:
        result = parse_atom()
        while peek() == "**":
            consume()
            right = parse_atom()
            result = math.pow(result, right)
        return result

    def parse_atom() -> float:
        token = peek()
        if token == "(":
            consume()
            result = parse_expression()
            consume()  # closing ')'
            return result
        if token == "-":
            consume()
            return -parse_atom()
        return float(consume())

    return parse_expression()


def _calculate(expression: str) -> str:
    """Compute a simple arithmetic expression safely."""
    cleaned = expression.strip()
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.\^]+$', cleaned):
        return f"Error: cannot compute '{cleaned}' — only basic arithmetic is supported"
    cleaned = cleaned.replace("^", "**")
    try:
        result = _parse_arithmetic(cleaned)
        return str(result)
    except Exception as error:
        return f"Error: {error}"


def _lookup(topic: str) -> str:
    """Mock knowledge lookup — returns canned responses for demo purposes."""
    knowledge_base: dict[str, str] = {
        "python": "Python is a high-level programming language created by Guido van Rossum, first released in 1991.",
        "neural network": "A neural network is a computing system inspired by biological neurons, consisting of layers of interconnected nodes that learn patterns from data.",
        "machine learning": "Machine learning is a subset of AI where systems learn from data to improve performance on tasks without being explicitly programmed.",
        "transformer": "The Transformer architecture (Vaswani et al., 2017) uses self-attention mechanisms to process sequences in parallel, enabling modern LLMs.",
        "pytorch": "PyTorch is an open-source deep learning framework developed by Meta AI, known for its dynamic computation graph and Pythonic API.",
        "react pattern": "ReAct (Reason + Act) is an agent pattern where an LLM alternates between reasoning about what to do and taking actions via tools.",
    }
    topic_lower = topic.strip().lower()
    for key, value in knowledge_base.items():
        if key in topic_lower:
            return value
    return f"No information found about '{topic}'. Try: {', '.join(knowledge_base.keys())}"


def _get_time(_input: str) -> str:
    """Return the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


# Available tools
TOOLS: list[Tool] = [
    Tool(name="calculate", description="Compute arithmetic expressions (e.g., '2 + 3 * 4')", function=_calculate),
    Tool(name="lookup", description="Look up information about a topic", function=_lookup),
    Tool(name="get_time", description="Get the current date and time", function=_get_time),
]

TOOL_MAP: dict[str, Tool] = {tool.name: tool for tool in TOOLS}


def _mock_llm_decide(question: str, history: list[dict[str, str]]) -> dict[str, str]:
    """Mock LLM: pattern-match to decide which tool to use.

    In a real agent, this would be an API call to an LLM with the system
    prompt, tool descriptions, and conversation history. The LLM returns
    structured output indicating its reasoning and chosen action.

    To swap in a real LLM client, replace this function body with:

        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            system="You are a ReAct agent with access to tools: ...",
            messages=history,
        )
        return parse_tool_call(response)
    """
    question_lower = question.lower()

    # If we already have observations, check whether we can answer
    if history:
        last = history[-1]
        if last.get("role") == "observation":
            return {
                "type": "answer",
                "reasoning": f"I have the information I need from the {last.get('tool', 'tool')} tool.",
                "content": f"Based on my research: {last['content']}",
            }

    # Route to tools based on keywords
    # Check for arithmetic: require at least one digit AND an operator
    math_match = re.search(r'\d[\d\s\+\-\*\/\(\)\.\^]*[\+\-\*\/\^][\d\s\+\-\*\/\(\)\.\^]*\d', question)
    if math_match:
        return {
            "type": "tool_call",
            "reasoning": "The question involves a calculation. I'll use the calculate tool.",
            "tool": "calculate",
            "input": math_match.group().strip(),
        }

    if any(word in question_lower for word in ["what is", "tell me about", "explain", "define", "who", "describe"]):
        for prefix in ["what is a ", "what is an ", "what is ", "tell me about ", "explain ", "define ", "describe "]:
            if prefix in question_lower:
                topic = question_lower.split(prefix, 1)[1].rstrip("?. ")
                return {
                    "type": "tool_call",
                    "reasoning": f"The user is asking about '{topic}'. I'll look it up.",
                    "tool": "lookup",
                    "input": topic,
                }

    if any(word in question_lower for word in ["time", "date", "today", "now", "clock"]):
        return {
            "type": "tool_call",
            "reasoning": "The user wants to know the current time.",
            "tool": "get_time",
            "input": "",
        }

    # Default: try lookup with the whole question
    return {
        "type": "tool_call",
        "reasoning": "I'm not sure what's being asked, so I'll try looking it up.",
        "tool": "lookup",
        "input": question,
    }


def _run_agent_loop(question: str, max_steps: int = 5) -> None:
    """Execute the ReAct loop for a given question."""
    console.print(Panel(f"[bold]Question:[/bold] {question}", border_style="cyan"))

    history: list[dict[str, str]] = []

    for step in range(1, max_steps + 1):
        console.print(f"\n[bold dim]─── Step {step} ───[/bold dim]")

        # Reason + Act
        decision = _mock_llm_decide(question, history)

        console.print(Panel(
            f"[italic]{decision['reasoning']}[/italic]",
            title="Thought",
            border_style="yellow",
        ))

        if decision["type"] == "answer":
            console.print(Panel(
                decision["content"],
                title="Answer",
                border_style="green",
            ))
            return

        # Execute tool
        tool_name = decision["tool"]
        tool_input = decision["input"]
        tool = TOOL_MAP.get(tool_name)

        if tool is None:
            console.print(f"[red]Unknown tool: {tool_name}[/red]")
            return

        console.print(f"  [cyan]Action:[/cyan] {tool_name}({tool_input!r})")
        result = tool.function(tool_input)
        console.print(Panel(result, title="Observation", border_style="blue"))

        history.append({
            "role": "observation",
            "tool": tool_name,
            "input": tool_input,
            "content": result,
        })

    console.print("[yellow]Reached maximum steps without a final answer.[/yellow]")


def run() -> None:
    """Run the agent on a set of example questions."""
    console.print(
        Panel(
            "[bold]Simple ReAct Agent[/bold]\n"
            "Reason → Act → Observe — the core loop of agentic AI.\n"
            "Uses a mock LLM (pattern matching) so no API key is needed.",
            border_style="bright_green",
        )
    )

    # Show available tools
    table = Table(title="Available Tools", show_header=True, header_style="bold green")
    table.add_column("Tool", style="cyan")
    table.add_column("Description")
    for tool in TOOLS:
        table.add_row(tool.name, tool.description)
    console.print(table)

    # Demo questions
    demo_questions = [
        "What is 42 * 17 + 3?",
        "What is a transformer?",
        "What time is it?",
    ]

    for question in demo_questions:
        console.print()
        _run_agent_loop(question)

    console.print(
        "\n[dim]This mock agent demonstrates the ReAct pattern: Reason about the question,"
        " Act by calling a tool, Observe the result, and repeat until answered."
        " Replace _mock_llm_decide with a real LLM call to make it intelligent.[/dim]\n"
    )
