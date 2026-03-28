from modelscope import snapshot_download

model_dir = snapshot_download("Qwen/Qwen3-0.6B")

print("模型下载完成: ", model_dir)