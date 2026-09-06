![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

# Connecting Tools with LLMs — Binding, Calling, Execution & Injected Arguments

## Table of Contents
- [Quick Recap: Creating Tools](#quick-recap-creating-tools)
- [Tool Binding](#tool-binding)
- [Tool Calling](#tool-calling)
- [Tool Execution](#tool-execution)
- [Args Dict vs. Full Tool Call → Tool Messages](#args-dict-vs-full-tool-call--tool-messages)
- [Sending Tool Messages Back to the LLM](#sending-tool-messages-back-to-the-llm)
- [InjectedToolArg](#injectedtoolarg)

---

## Quick Recap: Creating Tools

We already covered how to *create* tools — the three ways to turn a Python function into something LangChain can hand to an LLM:

1. **`@tool` decorator** — the fastest way. LangChain infers the input schema from your function's type hints, and the docstring becomes the tool's description.
2. **`StructuredTool` + Pydantic** — you explicitly define an input schema as a Pydantic model, giving you runtime validation and field-level descriptions. More strict than the decorator.
3. **`BaseTool` subclass** — the most manual, most powerful option. Every tool (including the two above) ultimately inherits from `BaseTool`. Subclassing it directly gives you custom async logic, internal state, and callback/observability hooks.

That's all *tool creation* — building the object. Now we move to the next stage: **how do you actually connect a created tool to an LLM, and how does the LLM decide which one to use?** That's three distinct steps: **Tool Binding → Tool Calling → Tool Execution.**

---

## Tool Binding

**Tool Binding** is the step where you **register tools with a Language Model (LLM)** so that:

1. The LLM knows **what tools are available**
2. It knows **what each tool does** (via its description)
3. It knows **what input format to use** (via its schema)

In practice, this is a single line:

```python
llm_with_tools = llm.bind_tools([multiply, get_weather])
```

`bind_tools()` doesn't change the tool itself — it takes each tool's metadata (name, description, argument schema) and attaches it to the LLM as part of every request going forward. Under the hood, each tool gets converted into the JSON-schema-style structure the model provider expects (name, description, parameters/properties/required — the same schema shape covered in the earlier tools README). From this point on, every time you call `llm_with_tools.invoke(...)`, the model is given the full list of available tools alongside the conversation, so it can reason over *whether* one is needed and, if so, *which one and with what arguments.*

Binding is purely a **registration/setup step** — it doesn't call anything and doesn't decide anything by itself. It just makes the LLM *aware* of what it's allowed to reach for.

---

## Tool Calling

**Tool Calling** is the process where the **LLM (language model) decides**, during a conversation or task, that it needs to use a specific tool (function) — and generates a **structured output** with:

- the **name** of the tool
- and the **arguments** to call it with

**Example:**

> User: *"What's 8 multiplied by 7?"*

The LLM responds with a tool call, not a direct answer:

```json
{
  "tool": "multiply",
  "args": { "a": 8, "b": 7 }
}
```

> ⚠️ **The LLM does not actually run the tool** — it just *suggests* the tool and the input arguments. The actual execution is handled by LangChain or by you.

This is worth dwelling on, because it's the core thing to understand about the whole tool-calling paradigm: **the LLM never gains the ability to execute code.** It's still just a text/JSON-in, text/JSON-out system. Binding tools to it doesn't give it a runtime, network access, or database access.

So what's the actual point of connecting tools to an LLM if you still have to execute everything yourself? The advantage isn't that the LLM gains *execution* power — it's that the LLM gains **decision-making power**:

> The LLM gains the ability to decide — reliably, in a structured, parseable way — what should be executed and with what arguments, based on natural language.

That's the real superpower, and it's non-trivial. Consider the alternative *without* tool calling: you'd have to hardcode all the routing logic yourself — *"if the user asks about weather, call this function; if they ask about math, call that one; manually parse their sentence to pull out a city name."* That's brittle regex/keyword matching, and it breaks the moment a user phrases something even slightly unexpectedly ("what's it like outside in the city where the Eiffel Tower is?").

With tool calling, the LLM does that hard part for you:
- **Intent detection** — figuring out *whether* a tool is needed at all, and *which* one.
- **Argument extraction** — pulling structured values (`a=8, b=7`) out of free-form natural language.
- **Reasoning over indirect references** — resolving things like "the city where X was born" into an actual usable argument.

So the responsibility splits cleanly:
- **LLM's job:** decide *what* should happen and *with what inputs* → outputs a structured tool call.
- **Your job (or LangChain's):** actually *execute* that decision → run the real Python function.

This split is also a safety feature, not just an implementation detail — since you're the one executing, you get a checkpoint to validate arguments, enforce permissions, log the call, or refuse to run it, before anything real actually happens.

---

## Tool Execution

**Tool Execution** is the step where the **actual Python function (tool)** is run using the input arguments that the LLM suggested during tool calling.

In simpler words:

> 💬 **The LLM says:**
> *"Hey, call the `multiply` tool with a=8 and b=7."*
>
> ⚙️ **Tool Execution** is when *you* or *LangChain* actually run:
> ```python
> multiply(a=8, b=7)
> ```
> → and get the result: `56`

This is the third and final piece of the pipeline:

| Step | Who does it | What happens |
|---|---|---|
| **Tool Binding** | You (setup) | Register tools + their schemas with the LLM |
| **Tool Calling** | LLM | Decides which tool + generates arguments (structured output, not execution) |
| **Tool Execution** | You / LangChain | Actually runs the Python function with those arguments |

Once execution happens, you have a raw result (`56`). But that result is just sitting in your program — the LLM doesn't know about it yet. To let the LLM use that result (e.g. to phrase a final natural-language answer), you need to send it back. How you do that depends on *what* you invoke the tool with, which is the next section.

---

## Args Dict vs. Full Tool Call → Tool Messages

There are two different ways to invoke a LangChain tool object once you have the LLM's suggested call, and they return different things:

**1. Passing just the `args` dictionary:**

```python
multiply.invoke({"a": 8, "b": 7})
# 56
```

This gives you back the **raw result** of the function — just the plain output value. Fine if all you need is the number itself.

**2. Passing the entire tool call object** (the full structure the LLM generated — including `name`, `args`, and `id`):

```python
tool_call = {
    "name": "multiply",
    "args": {"a": 8, "b": 7},
    "id": "call_123",
    "type": "tool_call"
}

multiply.invoke(tool_call)
```

This does **not** give you back a bare `56`. Instead, it gives you back a **`ToolMessage`** — a distinct message type in LangChain's message schema, alongside `SystemMessage`, `HumanMessage`, and `AIMessage`.

A `ToolMessage` looks roughly like:

```python
ToolMessage(
    content="56",
    name="multiply",
    tool_call_id="call_123"
)
```

**What is a `ToolMessage`, and why does it matter?**

A `ToolMessage` is a message type that represents *the result of a tool execution*, formatted so it can be dropped straight back into the conversation history the LLM sees. Its key fields:
- **`content`** — the actual result of the tool call (as a string).
- **`tool_call_id`** — links this result back to the *specific* tool call the LLM made (important when the LLM requests multiple tool calls in one turn — this `id` is what lets everything get matched up correctly).
- **`name`** — which tool produced this result.

The reason this distinction matters: an LLM conversation isn't just "user says something, AI says something." Once tools are involved, the conversation needs a slot for *"here's what happened when we ran the thing you asked for."* `ToolMessage` is that slot.

---

## Sending Tool Messages Back to the LLM

Once you have a `ToolMessage`, the full loop looks like this:

```python
messages = [HumanMessage(content="What's 8 multiplied by 7?")]

ai_response = llm_with_tools.invoke(messages)     # AIMessage containing a tool_call
messages.append(ai_response)

tool_result = multiply.invoke(ai_response.tool_calls[0])   # -> ToolMessage
messages.append(tool_result)

final_response = llm_with_tools.invoke(messages)  # LLM reasons over the ToolMessage
print(final_response.content)
# "8 multiplied by 7 is 56."
```

So the message list being sent back to the LLM ends up containing **all four message types together**:

```
[HumanMessage, AIMessage (tool call), ToolMessage (result), ...]
```

The LLM reads this whole sequence — the original question, its own earlier decision to call a tool, and the tool's actual output — and uses all of it to reason about and produce a **final answer in plain natural language**. This is exactly how the LLM "knows" the answer is 56 without ever having computed it itself: it's just reading the `ToolMessage` you handed back and describing it in words.

This request → tool call → execute → feed result back → final answer cycle is the complete tool-calling loop, and it's exactly what agent frameworks automate for you under the hood.

---

## InjectedToolArg

Sometimes a tool needs an argument that **should not come from the LLM at all** — even though the LLM is technically capable of generating a value for it. This is where `InjectedToolArg` comes in.

### The problem

Imagine you're chaining two tools together:

1. **`get_conversion_rate(from_currency, to_currency)`** — returns a live conversion rate (e.g. USD → PKR).
2. **`convert(base_amount, conversion_rate)`** — multiplies the base amount by the conversion rate to get the final converted value.

The natural workflow is: run tool 1, take its output, and feed it as the `conversion_rate` input to tool 2.

But here's the catch — **the LLM can technically fill in `conversion_rate` itself**, because it's just another parameter in the tool's schema, and the model has *some* number in its training data for what a conversion rate might roughly look like. The problem: that value is very likely **outdated or simply wrong**, because the LLM's knowledge has a training cutoff and exchange rates change constantly. If you let the LLM guess this argument instead of using the real, freshly-fetched value from tool 1, you'll silently get an incorrect final answer — the LLM will confidently multiply by a stale or hallucinated rate instead of the actual current one you just retrieved.

So conceptually:
- `get_conversion_rate` tool → produces the *real* value (like a "conversion factor").
- That real value needs to flow **programmatically** from tool 1's output into tool 2's input.
- The LLM should **not** be the one deciding/generating that particular argument — it should only be responsible for the arguments that genuinely come from the user's request (e.g. `base_amount`, `from_currency`, `to_currency`).

### The solution: `InjectedToolArg`

`InjectedToolArg` lets you mark a specific parameter in a tool's function signature as **"hidden from the LLM."** That argument is:
- **Excluded from the schema** the LLM sees (via `bind_tools`) — so the model never even knows it needs to supply it, and never attempts to.
- **Filled in by your code at execution time**, after the LLM's tool call comes back — you inject the real, correct value yourself (e.g. the actual output of the previous tool call) before running the function.

```python
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated

@tool
def convert(
    base_amount: float,
    conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """Convert a base amount using a given conversion rate."""
    return base_amount * conversion_rate
```

Here, `base_amount` still comes from the LLM (it's a normal, user-driven value — "convert 100 dollars"). But `conversion_rate` is annotated as injected: the LLM's generated schema for this tool won't even include it as something it needs to fill in. Instead, your program is responsible for supplying it — typically with the real value just fetched from `get_conversion_rate`.

### The resulting workflow

1. User asks: *"Convert 100 USD to PKR."*
2. LLM calls `get_conversion_rate(from_currency="USD", to_currency="PKR")`.
3. **You execute** that tool → get the real, current rate (e.g. `278.5`).
4. LLM also calls `convert(base_amount=100)` — notice it does **not** try to supply `conversion_rate`, because `InjectedToolArg` hid that parameter from it.
5. **You** manually inject the real rate from step 3 into the call before executing: `convert.invoke({"base_amount": 100, "conversion_rate": 278.5})`.
6. The function correctly returns `27850`, using the *actual* fetched rate — not a stale or hallucinated one from the LLM.

### Why this matters

`InjectedToolArg` draws a clean boundary between:
- **Arguments that should come from natural language / the user's intent** (the LLM is genuinely good at extracting these).
- **Arguments that should come from your program's own logic or from another tool's output** (the LLM has no business guessing these — it either doesn't have the real-time data, or the value must flow deterministically from code you control, not from the model's judgement).

This is especially important in **multi-tool / chained-tool workflows**, where the output of one tool is meant to feed directly into another. Without `InjectedToolArg`, you'd be relying on the LLM to either regenerate that intermediate value (risking incorrect/stale data) or you'd need extra prompt engineering to convince it not to guess. With it, the constraint is enforced structurally — the LLM literally never sees that field as something it's responsible for, so there's no ambiguity and no risk of it substituting its own (wrong) guess.