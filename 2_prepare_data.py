from datasets import load_dataset
from transformers import AutoTokenizer
from modelscope import snapshot_download

def main():
    dataset = load_dataset("json", data_files="data/train.jsonl")

    model_path = snapshot_download("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    def format_example(example):
        thought = f"<think>\n分析用户输入：{example['instruction']}。\n根据设定，我是图书馆智能助手厦小嘉。\n</think>\n"
        messages = [
            {"role": "system", "content": "你是一个图书馆智能助手，你的名字叫厦小嘉。"},
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": thought + example['output']}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {
            "text": text
        }

    dataset = dataset.map(format_example)

    dataset.save_to_disk("data/processed")

    print("数据处理完成")

if __name__ == "__main__":
    main()
