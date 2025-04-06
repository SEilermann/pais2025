from vllm import LLM
import os


os.environ["NCCL_SOCKET_IFNAME"] = "lo"

from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=2)
output = llm.generate("San Franciso is a")
