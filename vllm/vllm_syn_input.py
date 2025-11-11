# run_vllm_synth.py
import argparse
import csv
import json
import os
from pathlib import Path
from vllm import LLM, SamplingParams

# Map your friendly model names to local subpaths (adjust to your layout)
MODEL_SUBPATHS = {
    "Qwen2_5_72B": "Qwen2.5-72B",
    "DeepSeek_R1_Distill_Llama_70B": "DeepSeek-R1-Distill-Llama-70B",
    "Llama_3_1_8B": "Llama-3.1-8B",
    "Llama_3_3_70B_Instruct": "Llama-3.3-70B-Instruct",
    "gemma_3_27b_it": "gemma-3-27b-it",  # change if your folder name differs
}

def synthetic_prompts():
    # 10 synthetic prompts, each with a stable ID
    return [
        {"id": "P001", "prompt": "Explain photosynthesis in two sentences."},
        {"id": "P002", "prompt": "List 5 creative uses for a paperclip, one per line."},
        {"id": "P003", "prompt": "Give a plain-English summary of how a transformer model works."},
        {"id": "P004", "prompt": "Write a friendly email subject line to thank a mentor."},
        {"id": "P005", "prompt": "In one paragraph, compare TCP vs UDP for a beginner."},
        {"id": "P006", "prompt": "Provide a bullet list of 6 healthy breakfast ideas."},
        {"id": "P007", "prompt": "Explain the difference between accuracy and F1 score simply."},
        {"id": "P008", "prompt": "Draft a 3-sentence product blurb for a reusable water bottle."},
        {"id": "P009", "prompt": "Describe how to boil pasta perfectly in 5 steps."},
        {"id": "P010", "prompt": "Summarize the concept of opportunity cost with a short example."},
    ]

def resolve_model_path(args):
    # Special-case: if model_path points directly to actual weights folder for gemma
    if args.model == "gemma_3_27b_it":
        direct = Path(args.model_path)
        sub = Path(args.model_path) / MODEL_SUBPATHS["gemma_3_27b_it"]
        return str(direct if direct.is_dir() and not sub.is_dir() else sub)
    return str(Path(args.model_path) / MODEL_SUBPATHS[args.model])

def build_llm(args) -> LLM:
    specific_model_path = resolve_model_path(args)
    print(f"Loading model once from: {specific_model_path}")
    llm = LLM(
        model=specific_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        # Fresh: one prompt at a time, no shared in-flight context
        max_num_seqs=1,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
    )
    return llm

def build_sampling_params(args) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

def save_text_per_prompt(output_dir: Path, stem: str, pid: str, prompt: str, text: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}_{pid}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"ID: {pid}\n")
        f.write(f"Input:\n{prompt}\n\n")
        f.write(f"Generation:\n{text}\n")
    return str(path)

def append_jsonl(jsonl_path: Path, pid: str, prompt: str, text: str):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": pid, "prompt": prompt, "output": text}, ensure_ascii=False) + "\n")

def append_assign_csv(csv_path: Path, pid: str, prompt: str, output_file: str):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["id", "prompt", "output_file"])
        writer.writerow([pid, prompt, output_file])

def main():
    parser = argparse.ArgumentParser(description="Load once, run 10 synthetic prompts (fresh, no history), save one output per prompt.")
    # Model & IO
    parser.add_argument('--model_path', type=str, default="/home/paisteam/projects/models")
    parser.add_argument('--model', type=str, default='gemma_3_27b_it', choices=list(MODEL_SUBPATHS.keys()))
    parser.add_argument('--output_dir', type=str, default='/home/paisteam/projects/experiments',
                        help='Top-level experiments folder.')
    parser.add_argument('--experiment_name', type=str, default='exp_001',
                        help='Subfolder under output_dir to store this run.')
    parser.add_argument('--output_file', type=str, default='synth_outputs.txt',
                        help='Stem for per-prompt files, e.g. synth_outputs_P001.txt')

    # Optional pairing artifacts (relative paths save inside the experiment subfolder)
    parser.add_argument('--pairs_jsonl', type=str, default=None,
                        help='Optional: write id/prompt/output to this JSONL (relative → inside experiment folder).')
    parser.add_argument('--assign_csv', type=str, default=None,
                        help='Optional: write id→prompt→output_file mapping CSV (relative → inside experiment folder).')

    # Parallelism intentionally disabled to keep prompts fresh & isolated
    parser.add_argument('--tensor_parallel_size', type=int, default=2)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.80)
    parser.add_argument('--max_model_len', type=int, default=3000)

    # Sampling
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--max_tokens', type=int, default=2000)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    args = parser.parse_args()

    # Build experiment directory: <output_dir>/<experiment_name>/
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")

    # Resolve optional artifact paths
    def in_experiment_dir(p: str | None) -> Path | None:
        if not p:
            return None
        pth = Path(p)
        return experiment_dir / pth if not pth.is_absolute() else pth

    jsonl_path = in_experiment_dir(args.pairs_jsonl)
    csv_path = in_experiment_dir(args.assign_csv)

    prompts = synthetic_prompts()
    llm = build_llm(args)
    sampling_params = build_sampling_params(args)

    stem = Path(args.output_file).stem

    for i, item in enumerate(prompts, start=1):
        pid = item["id"]
        prompt = item["prompt"]

        # Fresh, stateless call: one prompt at a time
        outs = llm.generate(prompt, sampling_params)
        text = outs[0].outputs[0].text.strip()

        out_path = save_text_per_prompt(experiment_dir, stem, pid, prompt, text)
        print(f"[{i}/10] {pid} → saved to {out_path}")

        if jsonl_path:
            append_jsonl(jsonl_path, pid, prompt, text)
        if csv_path:
            append_assign_csv(csv_path, pid, prompt, out_path)

    print("Done.")

if __name__ == "__main__":
    main()

