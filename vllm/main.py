import argparse
from generation_files_py.Qwen2_5_72B_generation import main_Qwen2_5_72B
from generation_files_py.DeepSeek_R1_Distill_Llama_70B_generation import main_DeepSeek_R1_Distill_Llama_70B

def main():
    parser = argparse.ArgumentParser(description="Run generation with Qwen-style prompt and save to file.")

    # Model & output settings
    parser.add_argument('--model_path', type=str, default="/home/paisteam/models", help='Path to the model folder or HF repo')
    parser.add_argument('--output_dir', type=str, default='/home/paisteam/experiments', help='Directory to save output file')
    parser.add_argument('--output_file', type=str, default='qwen_output.txt', help='Output filename')

    #Model
    parser.add_argument('--model', type=str, default='Qwen2_5_72B', help='used model name')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='used model name')
    parser.add_argument('--max_model_len', type=int, default=2048, help='used model name')

    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--max_tokens', type=int, default=1000)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    args = parser.parse_args()

    if args.model=='Qwen2_5_72B':
        main_Qwen2_5_72B(args)
    
    if args.model=='DeepSeek_R1_Distill_Llama_70B':
        main_DeepSeek_R1_Distill_Llama_70B(args)



if __name__ == "__main__":
    main()

