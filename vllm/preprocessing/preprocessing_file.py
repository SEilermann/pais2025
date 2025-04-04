from vllm import LLM
import argparse
import os
import gc
import torch
from huggingface_hub import snapshot_download

# Helper function to free GPU memory
def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# 1. Load the OpenChat model
def main():
    parser = argparse.ArgumentParser(description="Run an experiment with an Ollama model and optionally save the response.")
    parser.add_argument('--openchat_llm', action='store_true', help='Download openchat_llm')
    parser.add_argument('--llama_3_1_8B_llm', action='store_true', help='Download llama_3_1_8B_llm')
    parser.add_argument('--Llama_3_3_70B_Instruct', action='store_true', help='Download Llama_3_3_70B_Instruct')
    parser.add_argument('--DeepSeek_R1_Distill_Llama_70B', action='store_true', help='Download DeepSeek_R1_Distill_Llama_70B')
    parser.add_argument('--Qwen2_5_72B', action='store_true', help='Download Qwen2_5_72B')
    parser.add_argument('--Mixtral_8x7B_Instruct', action='store_true', help='Download Mixtral_8x7B_Instruct')

    parser.add_argument('--model_DIR', type=str, default='', help='Prompt to send to the model')
    args = parser.parse_args()

    # Define the folder path
    folder = os.path.join(args.model_DIR,"models")
    os.makedirs(folder, exist_ok=True)
    
    if args.openchat_llm:
        snapshot_download(
            repo_id="openchat/openchat_3.5",
            local_dir=os.path.join(folder,"openchat_3.5"),
            local_dir_use_symlinks=False)
    
        #openchat_llm = LLM(model=os.path.join(folder,"openchat_3.5"))
        # Unload OpenChat from GPU
        #unload_model(openchat_llm)
    if args.llama_3_1_8B_llm:
        snapshot_download(
            repo_id="meta-llama/Llama-3.1-8B",
            local_dir=os.path.join(folder,"Llama-3.1-8B"),
            local_dir_use_symlinks=False)

        #llama_3_1_8B_llm = LLM(model=os.path.join(folder,"Llama-3.1-8B"))
        # Unload OpenChat from GPU
        #unload_model(llama_3_1_8B_llm)

    #  LLaMA 3.3 70B
    if args.Llama_3_3_70B_Instruct:
        snapshot_download(
            repo_id="meta-llama/Llama-3.3-70B-Instruct",
            local_dir=os.path.join(folder, "Llama-3.3-70B-Instruct"),
            local_dir_use_symlinks=False
        )

    # DeepSeek-R1-Distill-Llama-70B
    if args.DeepSeek_R1_Distill_Llama_70B:
        snapshot_download(
            repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            local_dir=os.path.join(folder, "DeepSeek-R1-Distill-Llama-70B"),
            local_dir_use_symlinks=False
        )

    # Qwen2.5-72B
    if args.Qwen2_5_72B:
        snapshot_download(
            repo_id="Qwen/Qwen2.5-72B",
            local_dir=os.path.join(folder, "Qwen2.5-72B"),
            local_dir_use_symlinks=False
        )

    # Mixtral-8x7B-Instruct
    if args.Mixtral_8x7B_Instruct:
        snapshot_download(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            local_dir=os.path.join(folder, "Mixtral-8x7B-Instruct-v0.1"),
            local_dir_use_symlinks=False
        )


    
if __name__ == "__main__":
    main()