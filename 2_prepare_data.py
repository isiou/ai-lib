from datasets import load_dataset
from transformers import AutoTokenizer
from modelscope import snapshot_download


def main():
    dataset = load_dataset("json", data_files="data/train.jsonl")

    model_path = snapshot_download("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def format_example(example):
        system_prompt = """你的名字是厦小嘉，是厦门大学嘉庚学院图书馆的智能助手，用于帮助用户解决相关咨询问题等。
        规则一：如果【参考资料】中给出了与问题相关的明确信息，必须严格根据参考资料准确地回答。
        规则二：如果【参考资料】中无相关资料或无法解答该问题时，且问题是关于图书馆业务（如借书、预约、开馆时间等）时，请运用你作为图书馆助手的常识给出通用的解答。
        规则三：如果用户提出问题与图书馆业务无关时，请不要自作主张进行回答。
        规则四：涉及政治、意识形态、色情等敏感信息时，拒绝回答！"""

        context = example.get("input", "")
        if not context:
            context = "无匹配的参考资料"

        user_prompt = (
            f"【参考资料】\n{context}\n\n【用户问题】\n{example['instruction']}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
