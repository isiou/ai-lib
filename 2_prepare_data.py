from datasets import load_dataset
from transformers import AutoTokenizer
from modelscope import snapshot_download


def main():
    dataset = load_dataset("json", data_files="data/train.jsonl")

    model_path = snapshot_download("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def format_example(example):
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_example)

    dataset.save_to_disk("data/processed")

    print("数据处理完成")


if __name__ == "__main__":
    main()
