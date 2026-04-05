import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from modelscope import snapshot_download

model_path = snapshot_download("Qwen/Qwen3-0.6B")


def main():
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 启用 4-bit NF4 精度量化策略以压缩显存开销
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 在计算设备上加载量化后的模型网络
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 对预训练模型进行梯度检查点注入等量化前置准备
    model = prepare_model_for_kbit_training(model)

    # 设定 LoRA 旁路自适应层超参数
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # 读取数据集
    dataset = load_from_disk("data/processed")

    # 验证集
    # if "validation" not in dataset:
    # dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    # 配置训练器参数
    training_args = SFTConfig(
        # 输出路径
        output_dir="outputs/lora",
        # 每张显卡处理样本数
        per_device_train_batch_size=1,
        # 指定梯度后更新参数
        gradient_accumulation_steps=4,
        # 防止轮数太多导致过拟合
        num_train_epochs=5,
        learning_rate=1e-4,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        # 验证策略
        # eval_strategy="steps",
        # 验证步长
        # eval_steps=100,
        # 保留最佳
        # load_best_model_at_end=True,
        save_total_limit=3,
        report_to="none",
        dataset_text_field="text",
        max_length=2048,
        # 打包短序列
        # packing=True,
        # 并行处理
        dataset_num_proc=4,
    )

    # 挂载 SFTTrainer 进行标准有监督调优
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        args=training_args,
    )

    # 启动反向传播及权重更新
    trainer.train()

    # 落盘保存独立分离的 LoRA 微调权重
    model.save_pretrained("outputs/lora_adapter")
    tokenizer.save_pretrained("outputs/lora_adapter")

    print("LoRA 微调完成")


if __name__ == "__main__":
    main()
