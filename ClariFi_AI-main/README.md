# ClariFi AI: Financial Analysis Agent

ClariFi AI is an intelligent ReAct (Reasoning + Acting) agent designed to answer complex financial questions by retrieving and analyzing information from a corpus of financial documents. It leverages a local Large Language Model (LLM) to reason through queries, execute search actions, and synthesize final answers based on retrieved evidence.

## Project Overview

This project implements a complete agentic workflow from scratch, demonstrating the core principles of modern AI systems:
1.  **Information Retrieval**: A custom TF-IDF search engine to find relevant documents.
2.  **ReAct Prompting**: A structured prompting technique that interleaves reasoning ("Thought"), action execution ("Action"), and observation ("Observation").
3.  **Local LLM Integration**: Uses the `Qwen/Qwen2.5-0.5B-Instruct` model via Hugging Face Transformers for autonomous decision-making.
4.  **Dual Interface**: 
    *   **Web UI**: A modern, aesthetic Streamlit interface for interactive use.
    *   **CLI**: A terminal-based interface for quick testing and debugging.

## Codebase Structure

The project is organized into modular components:

### 1. `main.py` (Streamlit UI)
*   **Purpose**: The primary entry point for the web application.
*   **Features**:
    *   Interactive text input for queries.
    *   Real-time streaming of the agent's thought process.
    *   Expandable execution traces.
    *   Aesthetic design with custom CSS styling.

### 2. `cli_main.py` (Terminal Interface)
*   **Purpose**: The entry point for command-line execution.
*   **Features**:
    *   Runs a hardcoded sample query (configurable in code).
    *   Prints full logs to the console.

### 3. `agent_system.py`
*   **Purpose**: The brain of the operation.
*   **Key Components**:
    *   `ReActAgent`: A class that implements the agent loop.
    *   `run()`: The main method that orchestrates the cycle:
        1.  Build Prompt -> 2. Call LLM -> 3. Parse Action -> 4. Execute Tool -> 5. Repeat.

### 4. `knowledge_base.py`
*   **Purpose**: Acts as the database and search engine.
*   **Key Components**:
    *   `CORPUS`: A collection of financial documents (e.g., Nestle 2024 Report, XYZ Inc. Balance Sheet).
    *   `compute_tf`, `compute_df`, `tfidf_vector`: Implements the TF-IDF algorithm from first principles.
    *   `cosine`: Computes similarity between query and document vectors.
    *   `tool_search`: The tool function exposed to the agent.

### 5. `language_model.py`
*   **Purpose**: Wraps the Hugging Face model for easy inference.
*   **Key Components**:
    *   `hf_llm`: The main generation function. It takes a prompt and returns the model's next "Thought" and "Action".
    *   **Model Loading**: Automatically handles device selection (CPU vs. CUDA) and includes a fallback to `local_files_only` for robust offline/slow-network usage.
    *   **Streaming**: Supports `TextStreamer` for real-time output in both CLI and Streamlit.

### 6. `prompting_techniques.py`
*   **Purpose**: Manages the interface between the agent and the LLM.
*   **Key Components**:
    *   `parse_action`: Parses the LLM's output (e.g., `Action: search[query="..."]`) into executable function calls.
    *   `make_prompt`: Constructs the context window.

## Prerequisites

*   **Python**: 3.8 or higher.
*   **Hardware**:
    *   **CPU**: Sufficient for running the 0.5B model (slow but functional).
    *   **GPU (Optional)**: NVIDIA GPU with CUDA for faster inference.

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install Dependencies**:
    You need the Hugging Face ecosystem, PyTorch, and Streamlit.
    ```bash
    pip install torch transformers accelerate bitsandbytes streamlit
    ```
    *Note: `bitsandbytes` is optional but recommended for 8-bit loading if you have a compatible GPU.*

## Usage

### Option 1: Interactive Web UI (Recommended)
Run the Streamlit app to interact with the agent in your browser:

```powershell
streamlit run main.py
```

**Features**:
*   Enter any financial question.
*   Watch the agent "think" in real-time inside the expandable "Agent Thinking Process" section.
*   View the final answer in a highlighted success box.
*   Inspect the full JSON execution trace for debugging.

### Option 2: Command Line Interface
Run the agent in the terminal with a default sample query:

```powershell
python cli_main.py
```

## Configuration

To change the model or enable 8-bit loading, edit `language_model.py`:

```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Change to a larger model if you have the hardware
LOAD_8BIT = True  # Set to True if you have bitsandbytes and a GPU
```

## Troubleshooting

*   **"Hanging" during load**: The model download can be slow. The code now includes status prints to keep you informed. If it fails, it attempts to use cached local files.
*   **Slow Generation**: On a CPU, generating text is slow. The streaming feature is enabled in both UI and CLI so you can see progress character-by-character.
*   **`torch_dtype` Warning**: This has been fixed in the latest codebase by using the `dtype` argument.

## License
This project is for educational purposes as part of CS 4100.

