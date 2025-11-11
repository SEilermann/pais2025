# Preprocessing
Here some information to download some GenAI models from [huggingface](https://huggingface.co/models).

## Run preprocessing and 🔒 Notes
For Meta's LLaMA models, you must first accept their license on the Hugging Face model page.
You must be logged into Hugging Face (huggingface-cli login) before downloading gated models. So the preprocessing is:
1. Accept the licences at huggingface
    - [ ] Llama-3.3-70B-Instruct
    - [ ] Llama-3.1-8B
    - [ ]
    - [ ]
    
2. If you have no huggingface access token create it. After run the following command to login, therefore use your token.
```bash
huggingface-cli login
```

3. After you active `paisenv` run preprocessing script:
A) Download only specific models, therefore define `model_DIR` where a folder `models` will be generated with separat folder per model.
```bash
conda activate paisenv
python /home/paisteam/projects/test/vllm/preprocessing_file.py --model_DIR /home/paisteam --openchat_llm --llama_3_1_8B_llm
```

B) For all integrategated models run:
```bash
conda activate paisenv
python /beegfs/home/e/eilermas/Projekte/pais2025/preprocessing_file.py --model_DIR /beegfs/home/e/eilermas/Projekte/pais2025 --openchat_llm --llama_3_1_8B_llm --Llama_3_3_70B_Instruct --DeepSeek_R1_Distill_Llama_70B --Qwen2_5_72B --Mixtral_8x7B_Instruct
```
