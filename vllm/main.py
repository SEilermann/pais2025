import os
os.environ["NCCL_SOCKET_IFNAME"] = "lo"


import argparse
from generation_files_py.Qwen2_5_72B_generation import main_Qwen2_5_72B
from generation_files_py.DeepSeek_R1_Distill_Llama_70B_generation import main_DeepSeek_R1_Distill_Llama_70B
from generation_files_py.Llama_3_1_8B_generation import main_Llama_3_1_8B
from generation_files_py.Llama_3_3_70B_Instruct_generation import main_Llama_3_3_70B_Instruct
from generation_files_py.gemma_3_27b_it_generation import main_gemma_3_27b_it


def main():
    parser = argparse.ArgumentParser(description="Run generation with Qwen-style prompt and save to file.")

    # Model & output settings
    parser.add_argument('--model_path', type=str, default="/home/paisteam/projects/models", help='Path to the model folder or HF repo')
    parser.add_argument('--output_dir', type=str, default='/home/paisteam/projects/experiments', help='Directory to save output file')
    parser.add_argument('--output_file', type=str, default='gemma_3_27b_it_output.txt', help='Output filename')

    #Model
    parser.add_argument('--model', type=str, default='gemma_3_27b_it', help='used model name')
    parser.add_argument('--tensor_parallel_size', type=int, default=2, help='used model name')
    parser.add_argument('--max_model_len', type=int, default=100, help='used model name')

    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--max_tokens', type=int, default=20)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    args = parser.parse_args()

    


    if args.model=='Qwen2_5_72B':
        main_Qwen2_5_72B(args)
    
    elif args.model=='DeepSeek_R1_Distill_Llama_70B':
        main_DeepSeek_R1_Distill_Llama_70B(args)

    elif args.model=='Llama_3_1_8B':
        main_Llama_3_1_8B(args)

    elif args.model=='Llama_3_3_70B_Instruct':
        main_Llama_3_3_70B_Instruct(args)
    
    elif args.model=='gemma_3_27b_it':
          main_gemma_3_27b_it(args)

if __name__ == "__main__":
    main()

