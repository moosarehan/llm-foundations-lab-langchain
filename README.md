# LangChain Foundations Lab

This repository is a hands-on study guide for building applications with
[LangChain](https://www.langchain.com/). It is organized as a set of small,
focused modules rather than one production application. Each module combines
explanatory notes with Python examples that demonstrate one part of the
LangChain ecosystem: models, prompts, LCEL runnables, chains, tools, document
loading, text splitting, embeddings, vector stores, retrieval, structured
output, RAG, and agents.

The repository is intended to be read progressively. Start with prompts and
models, then move through composition and data handling before studying RAG,
tool calling, and agents.

## Table of Contents

- [What You Will Learn](#what-you-will-learn)
- [Repository Map](#repository-map)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Create a virtual environment](#2-create-a-virtual-environment)
  - [3. Activate the environment](#3-activate-the-environment)
  - [4. Install dependencies](#4-install-dependencies)
  - [5. Configure API keys](#5-configure-api-keys)
- [Running Examples](#running-examples)
- [Learning Path](#learning-path)
- [Module Guide](#module-guide)
- [Data and Notebooks](#data-and-notebooks)
- [Common Troubleshooting](#common-troubleshooting)
- [Security and Cost Notes](#security-and-cost-notes)
- [Keeping the Environment Reproducible](#keeping-the-environment-reproducible)
- [Contributing](#contributing)

## What You Will Learn

The examples are designed to make the main boundaries in a LangChain
application visible:

- How prompts and chat messages become model inputs.
- How models, parsers, retrievers, and tools share the `Runnable` interface.
- How LCEL composes steps with the pipe operator (`|`).
- How sequential, parallel, and conditional workflows differ.
- How documents move from files or websites into chunks, embeddings, and a
  vector store.
- How retrievers provide context for retrieval-augmented generation (RAG).
- How output parsers and provider-native structured output produce predictable
  application data.
- How an LLM can select a tool while your application remains responsible for
  executing and validating that tool call.
- How the ReAct pattern extends model calls with tools, context, and iterative
  action/observation steps.

## Repository Map

```text
.
|-- AI-agents/                Agent concepts and the ReAct pattern
|-- langchain-chains/         Sequential, parallel, and conditional chains
|-- langchain-chromadb/       Vector stores and Chroma
|-- langchain-docsloader/     Text, PDF, CSV, directory, and web loaders
|-- langchain-models/         Chat models, LLMs, and embeddings
|-- langchain-prompt/         Prompts, messages, placeholders, and chatbots
|-- langchain-retrievers/     Retriever concepts and a Jupyter notebook
|-- langchain-runnable/       LCEL runnable primitives
|-- langchain-structure-output Structured output and output parsers
|-- langchain-textsplitter/   Length, structure, document, and semantic splitting
|-- RAG/                      Retrieval-augmented generation concepts
|-- toolcalling/              Binding, calling, and executing tools
|-- tools/                    Built-in and custom LangChain tools
|-- requirements.txt          Shared Python dependencies
`-- .gitignore                Secret, cache, and local-environment exclusions
```

Each module README is the detailed lesson for that module. The Python files
are intentionally small and can be read alongside the corresponding lesson.

## Prerequisites

- Python 3.10 or newer is recommended.
- Git, if you are cloning the repository.
- A working internet connection for package installation and provider-backed
  examples.
- A Google AI Studio API key for Gemini examples.
- A Hugging Face access token for Hugging Face API examples.
- Enough disk space for local Hugging Face models if you run local embedding or
  local model examples. PyTorch and model downloads can be large.

Some examples are local and do not need an API key, such as prompt templates,
many runnable demonstrations, document loaders for included files, and the
conceptual lessons.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/moosarehan/llm-foundations-lab-langchain.git
cd llm-foundations-lab-langchain
```

If you already have the repository, open its root directory in VS Code and
continue with the virtual environment setup.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

macOS or Linux:

```bash
python3 -m venv .venv
```

The repository's local `.venv/` directory is ignored by Git and should remain
outside version control.

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

macOS or Linux:

```bash
source .venv/bin/activate
```

If PowerShell blocks activation for the current session, run PowerShell as your
normal user and use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

Upgrade packaging tools, then install the shared requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file includes LangChain integrations, Google Gemini,
Hugging Face and PyTorch, Chroma, document loaders, web utilities, and common
data-science packages. Installation may take time because PyTorch and
`sentence-transformers` can be large.

Confirm that the environment is being used:

```bash
python -c "import sys; print(sys.executable)"
python -c "import langchain; print('LangChain import succeeded')"
```

### 5. Configure API keys

The examples use these environment variables:

| Variable | Used for | Where to get it |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini chat and model examples | [Google AI Studio](https://aistudio.google.com/apikey) |
| `HUGGINGFACEHUB_API_TOKEN` | Hugging Face hosted models and embeddings | [Hugging Face access tokens](https://huggingface.co/settings/tokens) |

Copy the template for the module you are running. For example, from the
repository root:

```powershell
Copy-Item langchain-models\.env.example langchain-models\.env
```

Then edit `langchain-models/.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

The same pattern applies to modules that contain their own `.env.example`:
`langchain-chains`, `langchain-chromadb`, `langchain-models`,
`langchain-prompt`, `langchain-runnable`, `langchain-structure-output`, and
`langchain-textsplitter`.

Some scripts load `.env` relative to the current module, so run them from the
repository root using the paths shown below, or follow the module README when
it specifies a different working directory. Never commit real credentials;
`.env` files are ignored by `.gitignore`.

## Running Examples

Run commands from the repository root with the virtual environment activated.
Python filenames containing hyphens are valid as script paths, but cannot be
imported as ordinary Python module names.

### Prompts and chat messages

```bash
python langchain-prompt/chatprompttemplate.py
python langchain-prompt/messageplaceholder.py
python langchain-prompt/messages.py
python langchain-prompt/prompt-generate.py
```

Read [langchain-prompt/README.md](langchain-prompt/README.md) for the complete
prompt and message walkthrough. The chatbot and UI examples may require a
configured provider and additional local runtime setup.

### Runnables and LCEL

```bash
python langchain-runnable/runnable-sequence.py
python langchain-runnable/runnable-parallel.py
python langchain-runnable/runnable-branch.py
python langchain-runnable/runnable-lambda.py
python langchain-runnable/runnable-passthrough.py
```

These examples demonstrate the common `invoke`-based interface and composition
patterns. See [langchain-runnable/README.md](langchain-runnable/README.md).

### Chains

```bash
python langchain-chains/sequentialchain.py
python langchain-chains/parallelchains.py
python langchain-chains/conditional_chains.py
```

These examples generally need the keys configured in
`langchain-chains/.env`. See [langchain-chains/README.md](langchain-chains/README.md).

### Models and embeddings

```bash
python langchain-models/ChatModels/chat-gemini.py
python langchain-models/ChatModels/chat-huggingface.py
python langchain-models/ChatModels/huggingface-local.py
python langchain-models/Embeddings/document-similar.py
python langchain-models/Embeddings/hf-api-docs.py
python langchain-models/Embeddings/hf-api-embedding.py
python langchain-models/Embeddings/hf-local-embed.py
python langchain-models/LLM/llm-demo.py
```

Hosted examples need the relevant provider credentials. Local examples may
download model weights on their first run. The model-specific guide is
[langchain-models/README.md](langchain-models/README.md).

### Document loaders and text splitters

```bash
python langchain-docsloader/textloader.py
python langchain-docsloader/pypdfloader.py
python langchain-docsloader/csvloader.py
python langchain-docsloader/directoryloader.py
python langchain-docsloader/webloader.py

python langchain-textsplitter/lengthbased.py
python langchain-textsplitter/textstructurebased.py
python langchain-textsplitter/documentbased.py
python langchain-textsplitter/semanticbased.py
```

The loader examples use included fixtures such as `clean.txt`, `user.csv`,
`artificial_intelligence.pdf`, and PDFs in `langchain-docsloader/books/`.
Web loading requires network access and should be used respectfully. See the
[document loader guide](langchain-docsloader/README.md) and
[text splitter guide](langchain-textsplitter/README.md).

### Structured output

```bash
python langchain-structure-output/str_outputparser.py
python langchain-structure-output/json_outputparser.py
python langchain-structure-output/structure_outputparser.py
python langchain-structure-output/pydantic_outputparser.py
python langchain-structure-output/with-structure-output-typedict.py
python langchain-structure-output/with-structure-output-pydantic.py
python langchain-structure-output/with-structure-output-json.py
```

The parser examples illustrate increasing levels of output structure and
validation. Provider-backed examples require the appropriate module `.env`.
See [langchain-structure-output/README.md](langchain-structure-output/README.md).

### Vector stores, retrievers, tools, and agents

```bash
python langchain-chromadb/chromavectorestore.py
python toolcalling/currencyconversion.py
```

The retriever and tools directories also include Jupyter notebooks and detailed
conceptual guides. `AI-agents/` currently contains the agent and ReAct lesson;
it does not yet contain a standalone Python agent script.

## Learning Path

For a coherent progression, use this order:

1. **Prompts and messages**: Learn how model inputs are represented.
2. **Models and embeddings**: Compare hosted and local model integrations.
3. **Runnables**: Learn the shared interface and LCEL composition.
4. **Chains**: Apply sequence, parallel, and conditional control flow.
5. **Output structure**: Turn model responses into usable application data.
6. **Document loaders**: Convert files and web pages into `Document` objects.
7. **Text splitters**: Prepare documents for retrieval and embedding.
8. **Vector stores and retrievers**: Index and search semantic representations.
9. **RAG**: Combine retrieved context with generation.
10. **Tools, tool calling, and agents**: Let models choose and coordinate actions.

## Module Guide

| Module | Focus | Main entry points |
|---|---|---|
| [AI agents](AI-agents/README.md) | Agent characteristics and the ReAct pattern | Lesson README |
| [Chains](langchain-chains/README.md) | Sequential, parallel, and conditional workflows | `sequentialchain.py`, `parallelchains.py`, `conditional_chains.py` |
| [Chroma](langchain-chromadb/README.md) | Vector stores, semantic search, and Chroma | `chromavectorestore.py` |
| [Document loaders](langchain-docsloader/README.md) | Loading text, PDF, CSV, directories, and web pages | `textloader.py`, `pypdfloader.py`, `csvloader.py`, `directoryloader.py`, `webloader.py` |
| [Models](langchain-models/README.md) | Chat models, LLMs, and embeddings | `ChatModels/`, `Embeddings/`, `LLM/` |
| [Prompts](langchain-prompt/README.md) | Prompt templates, messages, history, and chat flows | `chatprompttemplate.py`, `messageplaceholder.py` |
| [Retrievers](langchain-retrievers/README.md) | Retriever types and retrieval strategies | `langchain_retrievers (1).ipynb` |
| [Runnables](langchain-runnable/README.md) | Runnable primitives and LCEL | `runnable-*.py` |
| [Structured output](langchain-structure-output/README.md) | Parsers, schemas, TypedDict, and Pydantic | `*_outputparser.py`, `with-structure-output-*.py` |
| [Text splitters](langchain-textsplitter/README.md) | Length, structural, document, and semantic splitting | `lengthbased.py`, `textstructurebased.py`, `documentbased.py`, `semanticbased.py` |
| [RAG](RAG/README.md) | Retrieval-augmented generation concepts | Lesson README |
| [Tool calling](toolcalling/README.md) | Binding, calling, execution, and tool messages | `currencyconversion.py` |
| [Tools](tools/README.md) | Built-in and custom tools | `tools_in_langchain.ipynb` |

## Data and Notebooks

The repository includes small learning fixtures so several examples can be
run without creating data first:

- `langchain-docsloader/clean.txt` and `user.csv` for text and CSV loading.
- `langchain-docsloader/artificial_intelligence.pdf` and the PDFs under
  `langchain-docsloader/books/` for PDF and directory loading.
- `langchain-textsplitter/artificial_intelligence1.pdf` for document splitting.
- `langchain-retrievers/langchain_retrievers (1).ipynb` for retriever study.
- `tools/tools_in_langchain.ipynb` for tool exploration.

Open notebooks in VS Code or Jupyter after selecting the repository's `.venv`
kernel. Notebook cells may have different provider or package requirements
from the adjacent scripts, so run cells incrementally and read their outputs.

## Common Troubleshooting

### `ModuleNotFoundError`

Confirm that the virtual environment is activated and install from the root:

```bash
python -m pip install -r requirements.txt
```

Use `python -m pip`, rather than a bare `pip`, to ensure packages are installed
into the interpreter that runs the script.

### Missing API key

Check that the `.env` file is in the module expected by the script, that the
variable name matches the table above, and that the script loads environment
variables before creating the model. Do not paste keys into source files.

### Local model download or memory errors

Local Hugging Face examples can download large weights and may require more
RAM, disk space, or compatible PyTorch support than a hosted example. Start
with an API-backed or embedding example, then try local models after confirming
your machine has enough resources.

### PDF or web loader errors

Verify that the input path is correct relative to the directory from which the
script is launched. Web loaders also need network access; a remote site may
change, block automated requests, or return content that cannot be parsed.

### Provider or LangChain API changes

LangChain integrations evolve quickly. If an example raises an import or
constructor error after dependency installation, compare the installed package
version and consult the linked module README and the provider's current
documentation. Keep changes limited to the example being studied.

## Security and Cost Notes

- Treat API keys and Hugging Face tokens as passwords. Do not commit or share
  them.
- Review tool arguments before executing tools that access the network, files,
  shell commands, databases, or external services.
- Hosted model calls and web requests may incur cost or rate limits.
- Local model downloads consume disk space and may use substantial CPU, RAM, or
  GPU resources.
- The examples are educational. Add authentication, input validation,
  permission checks, timeouts, logging, and error handling before adapting
  them for production.

## Keeping the Environment Reproducible

The repository intentionally uses one shared `requirements.txt` for the
learning modules. After changing dependencies, verify imports in the active
environment and update that file deliberately. Do not add `.venv/`, `.env`,
`__pycache__/`, model caches, or generated vector-store data to commits.

Because dependencies are currently specified without pinned versions, a fresh
installation can receive newer LangChain provider APIs than the examples were
written against. For repeatable experiments, record the working environment
with:

```bash
python -m pip freeze > requirements-lock.txt
```

Keep a lock file separate from the shared learning requirements if you create
one for a specific experiment.

## Contributing

1. Pick the module that matches the concept you are adding or correcting.
2. Keep examples small and runnable from the repository root where practical.
3. Update the module README and this root README when the repository map or
   setup instructions change.
4. Never commit credentials, local virtual environments, generated caches, or
   private source documents.
5. Run the affected example, or document why it cannot be run without a
   provider key or external service.

This project is a learning lab: clear explanations, focused examples, and
honest prerequisites are more valuable than hiding complexity behind a large
abstraction.
