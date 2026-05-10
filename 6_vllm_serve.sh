python -m vllm.entrypoints.openai.api_server \
    --model outputs/qwen_full \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

python -m uvicorn backend.main:app --reload --port 8080