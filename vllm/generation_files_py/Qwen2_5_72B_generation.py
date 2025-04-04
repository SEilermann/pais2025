import argparse
import os
from vllm import LLM, SamplingParams

def main_Qwen2_5_72B(args):
    

    # Qwen-style prompt
    prompt = (
        "<|system|>\nYou are a helpful assistant.\n"
        "<|user|>\nWhat is King Kong?\n"
        "<|assistant|>"
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    # Load model
    args.specific_model_path = os.path.join(args.model_path,'Qwen2.5-72B')
    llm = LLM(model=args.specific_model_path,max_model_len=args.max_model_len)

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        imax_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty
    )

    # Run generation
    outputs = llm.generate(prompt, sampling_params)

    # Get the text
    generation = outputs[0].outputs[0].text.strip()

    # Save to file
    with open(output_path, "w") as f:
        f.write(f"Input:\n{prompt}\n\n")
        f.write(f"Generation:\n{generation}\n")

    print(f"Generation saved to {output_path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run generation with Qwen-style prompt and save to file.")

    # Model & output settings
    parser.add_argument('--model_path', type=str, default="/home/paisteam/models/Llama-3.1-8B", help='Path to the model folder or HF repo')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Directory to save output file')
    parser.add_argument('--output_file', type=str, default='qwen_output.txt', help='Output filename')

    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--max_tokens', type=int, default=1000)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    args = parser.parse_args()

    main_Qwen2_5_72B(args)
