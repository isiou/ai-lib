from datasets import load_dataset
from transformers import AutoTokenizer
from modelscope import snapshot_download

model_path = snapshot_download("Qwen/Qwen3-0.6B")


def format_example(example):
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 默认系统提示词
    with open("prompt/system.md", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # 无知识召回场景
    context = example.get("input", "")
    if not context:
        context = "无匹配的参考资料"

    # 构建提示词
    user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{example['instruction']}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": example["output"]},
    ]

    # 拼接对话
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def main():
    # 原始微调数据
    dataset = load_dataset("json", data_files="data/datasets/train.json")

    # 执行映射转化并将处理好的箭头数据持久化
    dataset = dataset.map(format_example)

    # 保存处理后的数据集
    dataset.save_to_disk("data/processed")

    print("数据处理完成")


if __name__ == "__main__":
    main()
