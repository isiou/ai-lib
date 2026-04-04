from datasets import load_dataset
from transformers import AutoTokenizer
from modelscope import snapshot_download


def main():
    dataset = load_dataset("json", data_files="data/train.jsonl")

    model_path = snapshot_download("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def format_example(example):
        system_prompt = """你的名字是厦小嘉，是厦门大学嘉庚学院图书馆的专属智能助手。你的任务是耐心、友好、自然地解答读者的疑问。
        规则一：当你参考了【参考资料】时，请将信息转化为自然、口语化的客服回复，不要机械地复述"根据参考资料"这类生硬的词汇。
        规则二：如果【参考资料】为空，但问题是常见的图书馆业务（如怎么借书、证件丢失等），请结合你作为图书馆助手的常识，给出合理的通用指导。
        规则三：遇到与图书馆完全无关的问题（如历史、天气、写代码等），请礼貌地婉拒，例如：“抱歉，我主要负责解答图书馆相关的业务问题哦。”
        规则四：涉及政治、色情等敏感话题，请直接拒绝回答。"""

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
