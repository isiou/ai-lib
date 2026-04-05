import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from modelscope import snapshot_download

base_model_path = snapshot_download("Qwen/Qwen3-0.6B")
lora_path = "outputs/lora_adapter"


# 合并权重
def main():
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # 采用更高精度的 fp16 方式加载主网络
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 动态挂载指定的 LoRA 模块树
    model = PeftModel.from_pretrained(model, lora_path)

    # 将额外参数矩阵硬合入并卸载外挂节点
    model = model.merge_and_unload()

    # 输出整合后的全量模型
    model.save_pretrained("outputs/qwen_full")
    tokenizer.save_pretrained("outputs/qwen_full")

    print("模型合并完成")


if __name__ == "__main__":
    main()
