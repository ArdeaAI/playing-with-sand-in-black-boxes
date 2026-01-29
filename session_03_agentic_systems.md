# Session 03: Agentic Systems

From neural networks that classify to systems that reason, act, and learn from their environment.

**Demo:**
- `uv run sand session03-agent` — mock ReAct agent loop

---

## What Is an Agentic System?

An agent is a system that perceives its environment, reasons about what to do, takes actions, and learns from the results. In the LLM era, the canonical definition is:

**Agent = LLM + Memory + Planning + Tool Use**

<details>
<summary><strong>The ReAct Pattern</strong></summary>

ReAct (Reason + Act) is the foundational pattern for LLM agents. The agent alternates between:

1. **Thought:** Reason about the current situation and what to do next
2. **Action:** Call a tool (search, calculate, API call, code execution, etc.)
3. **Observation:** Receive the tool's output
4. **Repeat** until the agent has enough information to answer

```
User: "What is the population of France divided by the area of Texas?"

Thought: I need two pieces of information. Let me look up France's population first.
Action: lookup("population of France")
Observation: France has a population of approximately 68 million.

Thought: Now I need the area of Texas.
Action: lookup("area of Texas")
Observation: Texas has an area of approximately 268,596 square miles.

Thought: Now I can calculate: 68,000,000 / 268,596
Action: calculate("68000000 / 268596")
Observation: 253.17

Thought: I have the answer.
Answer: The population of France divided by the area of Texas is approximately 253 people per square mile.
```

The key insight: by explicitly generating reasoning traces, the LLM can break complex problems into steps, handle errors, and update its plan mid-execution.

**Paper:** Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*.

</details>

<details>
<summary><strong>Tool Use</strong></summary>

Tools transform an LLM from a text generator into an actor that can affect the world. Common tool categories:

| Category | Examples | Why Needed |
|----------|----------|-----------|
| **Information retrieval** | Web search, database queries, file reading | LLMs have knowledge cutoffs and can hallucinate facts |
| **Computation** | Calculator, code interpreter, data analysis | LLMs are unreliable at arithmetic and precise computation |
| **External APIs** | Email, calendar, CRM, deployment pipelines | To take real-world actions |
| **Code execution** | Python sandbox, shell commands | For complex logic the LLM describes but shouldn't simulate |

The pattern is universal: the LLM generates a structured tool call (name + arguments), a runtime executes it, and the result is fed back as context.

Modern LLMs (Claude, GPT-4, Gemini) have native tool-use capabilities — they're trained to emit structured tool calls rather than just text.

</details>

<details>
<summary><strong>Memory</strong></summary>

Agents need memory to maintain context across interactions and learn from experience.

**Short-term memory:** The conversation context window. Modern LLMs can hold 100K–2M tokens, but this is still finite. For long tasks, agents need strategies to manage what stays in context.

**Long-term memory:** External storage (vector databases, key-value stores, file systems) that persists across conversations. The agent retrieves relevant memories using semantic search (RAG) or structured queries.

**Working memory patterns:**
- **Scratchpad:** The agent writes intermediate results to a buffer, summarizing as it goes
- **Episodic memory:** Store and retrieve past experiences (what worked, what didn't)
- **Semantic memory:** Structured knowledge about the domain (entity relationships, learned facts)

The distinction between short-term and long-term memory mirrors human cognition, and many agent frameworks explicitly model both.

</details>

---

## Building a Simple Agent — Code Walkthrough

Our `simple_agent.py` demo implements the core ReAct loop with a mock LLM.

<details>
<summary><strong>Architecture</strong></summary>

```
┌─────────────────────────────────────────────┐
│                   Agent                      │
│                                              │
│   ┌───────────┐    ┌──────────────────────┐  │
│   │ Mock LLM  │───→│  Decision            │  │
│   │ (router)  │    │  - type: tool_call   │  │
│   └───────────┘    │    or answer          │  │
│                    │  - reasoning          │  │
│                    │  - tool + input       │  │
│                    └──────────────────────┘  │
│                           │                  │
│   ┌───────────────────────┼─────────────┐    │
│   │         Tool Registry               │    │
│   │  ┌───────────┐ ┌────────┐ ┌───────┐ │    │
│   │  │ calculate │ │ lookup │ │ time  │ │    │
│   │  └───────────┘ └────────┘ └───────┘ │    │
│   └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

**Components:**
- `Tool` dataclass: name, description, callable function
- `_mock_llm_decide()`: Pattern-matching router (stands in for a real LLM)
- `_run_agent_loop()`: The Reason → Act → Observe cycle
- History list: Tracks observations for multi-step reasoning

</details>

<details>
<summary><strong>From Mock to Real: What Would Change</strong></summary>

To convert this mock agent into a real one, you'd replace `_mock_llm_decide()` with an actual LLM API call:

```python
import anthropic

client = anthropic.Anthropic()

def _real_llm_decide(question, history):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system="You are a ReAct agent. Available tools: ...",
        messages=format_history(question, history),
        tools=format_tools(TOOLS),
    )
    return parse_response(response)
```

Everything else — the tool registry, the loop, the observation handling — stays the same. The architecture is the same; only the decision-making intelligence changes.

This is the key insight of agentic design: **the scaffolding is simple; the intelligence comes from the LLM.**

</details>

---

## Popular Agentic SaaS and Frameworks

<details>
<summary><strong>Foundation Model Providers (with agent capabilities)</strong></summary>

**Anthropic (Claude)**
- Tool use via structured `tool_use` blocks in the Messages API
- Extended thinking for complex multi-step reasoning
- Computer use capabilities (controlling desktop applications)
- Claude Code: an agentic coding assistant that runs in the terminal

**OpenAI (GPT-4, o-series)**
- Function calling API for structured tool use
- Assistants API with built-in code interpreter, file search, and function calling
- Agents SDK for building multi-agent workflows

**Google (Gemini)**
- Function calling with structured outputs
- Grounding with Google Search
- Agent Development Kit (ADK) for building multi-step agents
- AlphaEvolve: evolutionary algorithm discovery agent

</details>

<details>
<summary><strong>Agent Frameworks</strong></summary>

**LangChain / LangGraph**
- The most widely adopted framework for building LLM applications
- LangChain provides abstractions for chains (sequential LLM calls), tools, and memory
- LangGraph adds stateful, multi-step agent workflows with explicit graph-based control flow
- Large ecosystem of integrations (vector stores, LLMs, tools)

**CrewAI**
- Multi-agent framework where you define "crews" of specialized agents
- Each agent has a role, goal, and backstory
- Agents collaborate on tasks, delegating to each other
- Focus on making multi-agent systems accessible

**AutoGen (Microsoft)**
- Framework for building multi-agent conversations
- Agents can be LLMs, humans, or tools
- Supports complex conversation patterns (group chat, nested conversations)
- Strong emphasis on human-in-the-loop workflows

**Semantic Kernel (Microsoft)**
- SDK for integrating LLMs into applications
- Plugin architecture for tools and skills
- Planner that decomposes goals into plugin calls
- Enterprise-focused with .NET and Python SDKs

**Anthropic Agent SDK**
- Lightweight Python framework for building agents with Claude
- Model Context Protocol (MCP) for standardized tool integration
- Built-in support for tool use, multi-turn conversations, and agent handoffs
- Designed around Claude's native capabilities

</details>

<details>
<summary><strong>Specialized Agentic Tools</strong></summary>

**Coding Agents:**
- Claude Code (Anthropic) — terminal-based agentic coding
- GitHub Copilot / Copilot Workspace — IDE-integrated code generation and planning
- Cursor — AI-first IDE with agentic editing capabilities
- Devin (Cognition) — autonomous software engineering agent

**Research & Analysis:**
- Perplexity — search agent that synthesizes information from multiple sources
- Elicit — research assistant that finds, summarizes, and extracts data from papers

**Automation:**
- Zapier AI Actions — connect LLMs to 6000+ app integrations
- Relevance AI — build and deploy AI agents for business workflows

</details>

<details>
<summary><strong>Where Agents Are Heading</strong></summary>

The field is moving rapidly. Key trends:

1. **Computer use:** Agents that can see and interact with GUIs, not just APIs. Claude's computer use and OpenAI's Operator are early examples.

2. **Multi-agent systems:** Instead of one monolithic agent, teams of specialized agents that collaborate. One agent plans, another researches, a third writes code.

3. **Longer autonomy:** From single-turn tool calls to agents that work independently for hours or days, checking in with humans at key decision points.

4. **Better memory:** Moving beyond simple RAG to structured episodic and semantic memory that lets agents genuinely learn from experience.

5. **Standardized tooling:** Model Context Protocol (MCP) and similar standards are making tools portable across LLM providers, reducing vendor lock-in.

6. **Safety and alignment:** As agents gain more autonomy, the stakes of misalignment increase. Research into agent oversight, sandboxing, and human-in-the-loop controls is critical.

</details>

---

## References

### Papers

| Year | Authors | Title | Publication |
|------|---------|-------|-------------|
| 2023 | Yao et al. | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | *ICLR 2023* |

### Blog Posts & Surveys

- Lilian Weng (2023). [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — comprehensive survey of agent architectures, memory, planning, and tool use

### Frameworks & Documentation

- [Anthropic API — Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview)
- [Anthropic Agent SDK](https://github.com/anthropics/agent-sdk)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
