import os
from vllm import LLM, SamplingParams

def main_Qwen2_5_72B(args):
    # Qwen-style prompt
    prompt = (
        "What is King Kong?")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    # Load model
    args.specific_model_path = os.path.join(args.model_path,'Qwen2.5-72B')
    llm = LLM(model=args.specific_model_path,
              tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=0.80,
              max_num_seqs=1,
              max_model_len=args.max_model_len,
              disable_custom_all_reduce=True,  # eliminates warnings related to NCCL and P2P issues
            trust_remote_code=True)

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty
    )
    
    print(prompt)

    # Run generation
    outputs = llm.generate(prompt,
            sampling_params)

    # Get the text
    generation = outputs[0].outputs[0].text.strip()

    # Save to file
    with open(output_path, "w") as f:
        f.write(f"Input:\n{prompt}\n\n")
        f.write(f"Generation:\n{generation}\n")

    print(f"Generation saved to {output_path}")