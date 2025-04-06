from transformers import AutoModelForCausalLM, AutoTokenizer
import os

folder = '/beegfs/home/e/eilermas/Projekte/models'
reshard_path = os.path.join(folder, "test")
model_path = '/beegfs/home/e/eilermas/Projekte/models/gemma-3-27b-it_original'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="bfloat16"  # or "float16"
)

# 🔄 Save into new 8-way layout (change this if using 4 GPUs, etc.)
model.save_pretrained(reshard_path, max_shard_size="43GB")

# Also copy the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.save_pretrained(reshard_path)

