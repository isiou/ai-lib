import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from modelscope import snapshot_download

base_model_path = snapshot_download("Qwen/Qwen3-0.6B")
lora_path = "outputs/lora_adapter"

def main():

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(model, lora_path)

    # 合并LoRA
    model = model.merge_and_unload()

    model.save_pretrained("outputs/qwen_full")
    tokenizer.save_pretrained("outputs/qwen_full")

    print("模型合并完成")

if __name__ == "__main__":
    main()