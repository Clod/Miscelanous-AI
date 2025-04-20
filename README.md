# Project Notebooks Overview

This document provides a summary of the Jupyter Notebooks found in this directory, detailing their purpose, key technologies, and concepts demonstrated.

---

## 1. `GoogleAI_RAG_GeminiFlash_KDBAI.ipynb`

*   **Purpose**: Demonstrates a Retrieval-Augmented Generation (RAG) pipeline. It processes a PDF document, extracts text chunks, embeds them, stores them in a vector database (KDB.AI), and uses Google's Gemini Flash model to answer questions based on retrieved context.
*   **Technologies**:
    *   Google Generative AI (Gemini Flash 2.0)
    *   KDB.AI (Vector Database)
    *   Sentence Transformers (`all-MiniLM-L6-v2` for embeddings)
    *   `pdf2image`, `PyPDF2` (PDF processing)
    *   `asyncio`, `aiohttp` (Asynchronous operations)
    *   Pandas, NumPy
*   **Key Concepts**:
    *   Retrieval-Augmented Generation (RAG)
    *   Vector Databases & Similarity Search
    *   PDF Processing (OCR, Chunking)
    *   Text Embedding
    *   Asynchronous Programming

---

## 2. `HuggingFace_Agents_CodeExecution.ipynb`

*   **Purpose**: Introduces the Hugging Face `smolagents` library, focusing on the `CodeAgent` which executes Python code to accomplish tasks. Part of the Hugging Face Agents Course, using an "Alfred planning a party" theme.
*   **Technologies**:
    *   Hugging Face `smolagents` (`CodeAgent`, `HfApiModel`, `@tool`, `Tool`)
    *   Hugging Face Inference API/Endpoints (Models: Qwen, Phi-3)
    *   DuckDuckGoSearchTool, VisitWebpageTool
    *   Custom Tool Creation (`suggest_menu`, `catering_service_tool`, `SuperheroPartyThemeTool`)
    *   Hugging Face Hub (for sharing/loading agents)
    *   OpenTelemetry & Langfuse (for agent run tracing/observability)
    *   Python `datetime`
*   **Key Concepts**:
    *   Code Execution Agents
    *   Tool Definition and Usage
    *   Secure Code Execution (Authorized Imports)
    *   Agent Sharing and Reusability (Hub Integration)
    *   Agent Observability and Tracing

---

## 3. `HuggingFace_Agents_ToolCalling.ipynb`

*   **Purpose**: Compares the `ToolCallingAgent` with the `CodeAgent` from the previous notebook, illustrating how it uses structured tool calls rather than direct code execution. Part of the Hugging Face Agents Course.
*   **Technologies**:
    *   Hugging Face `smolagents` (`ToolCallingAgent`, `DuckDuckGoSearchTool`, `HfApiModel`)
*   **Key Concepts**:
    *   Tool Calling Agents
    *   Structured Tool Invocation vs. Code Execution

---

## 4. `HuggingFace_AgentTools_ToolCreationIntegration.ipynb`

*   **Purpose**: Deep dives into creating, sharing, and integrating various types of tools within the `smolagents` framework. Part of the Hugging Face Agents Course.
*   **Technologies**:
    *   Hugging Face `smolagents` (`@tool`, `Tool`, `CodeAgent`, `HfApiModel`, `load_tool`, `Tool.from_space`, `Tool.from_langchain`)
    *   Hugging Face Hub (Tool sharing)
    *   Gradio Client (for Space integration)
    *   LangChain (`load_tools`, `serpapi`)
    *   Google Search Results API (SerpApi)
*   **Key Concepts**:
    *   Custom Tool Creation (Decorator `@tool` vs. Class `Tool`)
    *   Tool Sharing and Loading (Hugging Face Hub)
    *   Integrating External Tools (Hugging Face Spaces, LangChain Tools)

---

## 5. `HuggingFace_MultiAgentSystems_HierarchicalTaskSolving.ipynb`

*   **Purpose**: Demonstrates building a multi-agent system with a hierarchical structure (manager/worker) to solve a complex task involving web research, data processing, and visualization. Part of the Hugging Face Agents Course.
*   **Technologies**:
    *   Hugging Face `smolagents` (`CodeAgent`, `HfApiModel`, `GoogleSearchTool`, `VisitWebpageTool`, `@tool`)
    *   Intended use of Together AI provider (commented out/replaced)
    *   OpenAI (`gpt-4o` for validation)
    *   Pandas, Matplotlib, Geopandas, Shapely, Plotly, Kaleido (Data handling and plotting)
*   **Key Concepts**:
    *   Multi-Agent Systems (MAS)
    *   Hierarchical Agent Architectures (Manager-Worker)
    *   Task Decomposition and Delegation
    *   Agent Collaboration
    *   Geospatial Data Visualization
    *   Automated Result Validation

---

## 6. `LiteLLM_Agent_ToolCalling.ipynb`

*   **Purpose**: Implements a simple agent loop from scratch using `litellm` to interact with LLMs (specifically OpenAI's GPT-4o) and perform basic tool calls based on parsed responses.
*   **Technologies**:
    *   `litellm` (LLM interaction abstraction)
    *   OpenAI (`gpt-4o`)
    *   Python `json`, `os`
*   **Key Concepts**:
    *   Manual Agent Loop Implementation
    *   Tool Calling (via JSON parsing)
    *   LLM Interaction Abstraction (`litellm`)
    *   Basic State Management (memory list)

---

## 7. `ONNXRuntime_OpticalMusicRecognition_OemerTool.ipynb`

*   **Purpose**: Showcases the use of the `oemer` tool for Optical Music Recognition (OMR), converting a music score image into MusicXML format, and then generating audio (MP3) and image previews using MuseScore.
*   **Technologies**:
    *   `oemer` (OMR tool, likely using ONNX Runtime internally)
    *   MuseScore 3 (Music notation software, used via CLI)
    *   Google Colab (`files.upload`, `IPython.display`)
    *   Matplotlib, OpenCV (`cv2`)
*   **Key Concepts**:
    *   Optical Music Recognition (OMR)
    *   MusicXML Standard
    *   Command-Line Interface (CLI) Tool Integration in Notebooks
    *   Image-to-Music Conversion
    *   Score-to-Audio/Image Generation

---
