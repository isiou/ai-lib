import subprocess
import sys
import time


def run_command(command, step_name):
    print(f"\n{'='*50}")
    print(f"▶ {step_name}")
    print(f"{'='*50}")

    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"{step_name} 完成")
    except subprocess.CalledProcessError as e:
        print(f"{step_name} 失败\n退出码: {e.returncode}")
        sys.exit(1)


def main():
    print("==================================================")
    print("    厦小嘉 RAG + LoRA 完整流水线运行脚本")
    print("==================================================")

    steps = [
        ("python 1_download_models.py", "[Step 1/6] 正在下载所需模型..."),
        ("python 2_prepare_data.py", "[Step 2/6] 正在格式化与处理微调训练集..."),
        ("python 3_train_lora.py", "[Step 3/6] 正在启动 LoRA 模型微调..."),
        ("python 4_merge_model.py", "[Step 4/6] 正在合并 LoRA 权重到基座模型..."),
        ("python 5_build_rag_db.py", "[Step 5/6] 正在构建知识库的 FAISS 向量数据库..."),
    ]

    for cmd, desc in steps:
        run_command(cmd, desc)

    # 启动服务
    step_6_desc = "[Step 6/6] 准备启动 vLLM 推理服务..."
    print(f"\n{'='*50}")
    print(f"{step_6_desc}")
    print("==================================================")
    print(" 注意：推理服务启动后将在此终端持续运行")
    print(" 服务就绪后打开新终端可使用 python 7_test_rag_api.py 测试效果")
    print(" 使用 Ctrl+C 停止服务")
    print("==================================================")
    time.sleep(3)

    try:
        # 使用 shell 脚本启动 vLLM
        subprocess.run("bash 6_vllm_serve.sh", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n服务已手动停止")
    except subprocess.CalledProcessError as e:
        print(f"vLLM 服务异常退出\n退出码: {e.returncode}")


if __name__ == "__main__":
    main()
