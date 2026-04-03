# Project Overview

This is a complete end-to-end Python project for fine-tuning, merging, deploying, and testing a Retrieval-Augmented Generation (RAG) system with a Large Language Model. The project acts as a "library smart assistant", leveraging a fine-tuned `Qwen/Qwen3-0.6B` model for text generation, `BAAI/bge-small-zh-v1.5` for document embeddings, and `BAAI/bge-reranker-base` for retrieving precision.

The pipeline handles downloading base models, preparing data, fine-tuning via LoRA (Low-Rank Adaptation), merging the adapter weights, building a FAISS vector database, serving the model via `vLLM`, and finally querying the RAG pipeline using a dual-stage retrieval approach (Vector Search -> Reranker).

# Key Technologies

*   **Models:** `Qwen/Qwen3-0.6B` (LLM), `BAAI/bge-small-zh-v1.5` (Embeddings), `BAAI/bge-reranker-base` (Cross-Encoder Reranker)
*   **Fine-tuning:** `peft` (LoRA), `trl` (SFTTrainer), `bitsandbytes` (4-bit quantization)
*   **Vector Database:** `faiss`, `sentence-transformers`
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

### Phase 5: Client API (Retrieval + Rerank + LLM Inference)
7.  `7_test_rag_api.py`: A client-side script that executes the complete query pipeline:
    *   **Retrieval:** Uses the vector database (FAISS) to fetch the top-K relevant chunks via embedding similarity.
    *   **Reranking:** Passes the retrieved chunks to the `bge-reranker-base` model to re-score and select the highest quality context.
    *   **Inference:** Prompts the locally running vLLM server with RAG-enhanced queries using the OpenAI client API.

# File Structure

*   `data/`: Contains raw (`train.jsonl`, `knowledge_base/`), processed, and vector (`rag/`) datasets.
*   `outputs/`: Stores the resulting model weights during fine-tuning and merging.

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
7.  **Test RAG API:** `python 7_test_rag_api.py`

# Development Conventions

*   **Environment:** Relies on heavily optimized ML environments (PyTorch, Triton, xformers, vLLM).
*   **Inference:** Uses the OpenAI API compatibility layer provided by `vLLM` for flexible LLM client integrations.
*   **Formatting:** Clean code structure utilizing `if __name__ == "__main__":` blocks for script execution.