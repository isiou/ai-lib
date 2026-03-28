import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from modelscope import snapshot_download

model_path =  snapshot_download("Qwen/Qwen3-0.6B")

def main():

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    # LoRA配置（适配8GB显存）
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    dataset = load_from_disk("data/processed")

    training_args = SFTConfig(
        output_dir="outputs/lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=100,
        learning_rate=5e-4,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        save_total_limit=2,
        report_to="none",
        dataset_text_field="text",
        max_length=1024
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()

    model.save_pretrained("outputs/lora_adapter")
    tokenizer.save_pretrained("outputs/lora_adapter")

    print("LoRA 微调完成")

if __name__ == "__main__":
    main()