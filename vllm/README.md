# VLLM HSU

## Requirements:
- Option to login to HSUper
- Huggingface account with a token for downloads

## Env. building
maybe neccessary on HSUper first

1. Load the `anaconda` module:
```bash
module load anaconda3/2021.11
```

2. Create the environment based on already defined environment `paisenv`:
```bash
conda env create -f paisenv.yml
```

## Preprocessing Download GenAI Models from Huggingface
3. Due to the fact that no internet is available on computing nodes start from the login node the following. For more information go to subfolder [preprocessing](./preprocessing).
At the moment the folliwng models are integrated and can be loaded:
- openchat-llm
- llama-3-1-8B-llm 
- Llama-3-3-70B-Instruct 
- DeepSeek-R1-Distill-Llama-70B
- Qwen2-5-72B
- Mixtral-8x7B-Instruct


## Running

```bash

```

### Run on HSUper
```bash
sbatch run.sh
```
