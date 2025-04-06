from vllm import LLM, SamplingParams

'''
Script to run it on ISCC or local computer.
'''

# 1. Load the OpenChat model

llm = LLM(model="/home/paisteam/models/Llama-3.1-8B",max_model_len=4000,tensor_parallel_size=2)


# 2. Define your prompt
#prompt = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nWhat's the capital of France?<|end|>\n<|assistant|>"
prompt = "What is King Kong?"
# 3. Set sampling parameters (adjust as needed)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)

# 4. Generate
outputs = llm.generate(prompt, sampling_params)

# 5. Print the result
for output in outputs:
    print(output.outputs[0].text)
