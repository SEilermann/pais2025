from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model and target folder
model_id = "meta-llama/Llama-3.3-70B-Instruct"
save_path = "/beegfs/home/e/eilermas/Projekte/models/Llama-3.3-70B-Instruct"

# Load model with appropriate dtype
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="bfloat16"  # or "float16" if you prefer
)

# Save the model in 8 shards (sized for 48GB VRAM)
model.save_pretrained(save_path, max_shard_size="48GB")

# Save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(save_path)

print("✅ Model and tokenizer saved to:", save_path)

