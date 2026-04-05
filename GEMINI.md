# Project Overview

This is a complete end-to-end Python project for fine-tuning, merging, deploying, and testing a Retrieval-Augmented Generation (RAG) system with a Large Language Model. The project acts as a "library smart assistant" (厦小嘉), leveraging a fine-tuned `Qwen/Qwen3-0.6B` model for text generation, `BAAI/bge-small-zh-v1.5` for document embeddings, and `BAAI/bge-reranker-base` for retrieving precision.

The pipeline handles downloading base models, preparing data, fine-tuning via LoRA (Low-Rank Adaptation), merging the adapter weights, building a FAISS vector database, serving the model via `vLLM`, and finally querying the RAG pipeline using a dual-stage retrieval approach (Vector Search -> Reranker) combined with local Function Calling (Tool Use) for dynamic database lookups.

# Key Technologies

*   **Models:** `Qwen/Qwen3-0.6B` (LLM), `BAAI/bge-small-zh-v1.5` (Embeddings), `BAAI/bge-reranker-base` (Cross-Encoder Reranker)
*   **Fine-tuning:** `peft` (LoRA), `trl` (SFTTrainer), `bitsandbytes` (4-bit quantization)
*   **Databases:** `faiss` (Vector Search), `sqlite3` (Relational Book Inventory)
*   **Serving:** `vLLM`, `fastapi`, `openai` python client
*   **Data/Model Hub:** `huggingface/transformers`, `datasets`, `modelscope`

# Architecture & Execution Sequence

The project execution is logically organized into distinct phases:

### Phase 1: Environment & Model Preparation
1.  `1_download_models.py`: Downloads the main text generation model, the sentence embedding model, and the cross-encoder reranking model.
2.  `2_prepare_data.py`: Formats the raw training data (`data/train.jsonl`) applying the model's chat template and saves it to `data/processed`.

### Phase 2: LLM Fine-Tuning
3.  `3_train_lora.py`: Runs LoRA fine-tuning on the processed dataset and saves the adapter to `outputs/lora_adapter`.
4.  `4_merge_model.py`: Merges the LoRA adapter with the base model, saving the fully merged model to `outputs/qwen_full`.

### Phase 3: RAG Knowledge Base Construction
5.  `5_build_rag_db.py`: Parses the local knowledge base (`data/knowledge_base/`), computes initial embeddings, and builds a FAISS index (`data/rag/faiss_index.bin`).

### Phase 4: Model Serving
6.  `6_vllm_serve.sh`: Shell script to serve the fully merged LLM locally using `vLLM`.

### Phase 5: Client API (Retrieval + Rerank + Function Calling)
The backend API (`backend/main.py`) integrates FastAPI with the vLLM service, a FAISS vector database, and an SQLite relational database for library inventory.
*   **Retrieval & Reranking (`backend/services/chat.py`):** Uses FAISS to fetch the top-K relevant chunks, re-scores them via `bge-reranker-base`, and builds the context.
*   **Database Integration (`backend/core/db.py`):** An SQLite database (`data/library.db`) is automatically initialized at startup, holding dynamic book inventory data (locations, call numbers, availability status).
*   **Function Calling (`backend/services/tools.py`):** Tools like `recommend_books` and `query_book_info` execute SQL queries against the database. The system uses a specialized prompt with Few-Shot examples and manual XML tag extraction as a fallback to reliably enforce tool use on the 0.6B small-parameter model.

# File Structure

*   `data/`: Contains raw (`train.jsonl`, `knowledge_base/`), processed, vector (`rag/`) datasets, and the relational database (`library.db`).
*   `outputs/`: Stores the resulting model weights during fine-tuning and merging.
*   `backend/`: Contains the FastAPI application, database initialization, RAG querying, and function calling tools.
*   `tests/`: Unit and integration tests for APIs and internal tool logic.

# Building and Running

Ensure you have installed the required dependencies, primarily utilizing a CUDA-enabled GPU for training and serving:

```bash
pip install -r requirements.txt
```

Run the pipeline sequentially:

1.  **Download Models:** `python 1_download_models.py`
2.  **Prepare Data:** `python 2_prepare_data.py`
3.  **Train LoRA:** `python 3_train_lora.py`
4.  **Merge Model:** `python 4_merge_model.py`
5.  **Build RAG DB:** `python 5_build_rag_db.py`
6.  **Serve Model:** `bash 6_vllm_serve.sh` (Leave this running in a separate terminal)
7.  **Run Application Server:** `python -m uvicorn backend.main:app --reload` (or access via FastAPI)

# Development Conventions

*   **Virtual Environment:** **MANDATORY.** All project activities (installing dependencies, running scripts, serving the model, executing tests, and static analysis) MUST be performed within the active virtual environment (`.venv`). Do not use global Python environments.
*   **Environment:** Relies on heavily optimized ML environments (PyTorch, Triton, xformers, vLLM).
*   **Inference:** Uses the OpenAI API compatibility layer provided by `vLLM` for flexible LLM client integrations.
*   **Static Analysis:** Use `mypy` and `ruff` for strict typing and linting checks. Un-typed third-party libraries (e.g., modelscope, datasets) should be ignored in `mypy` configurations.
*   **Formatting:** Clean code structure utilizing `if __name__ == "__main__":` blocks for script execution.
