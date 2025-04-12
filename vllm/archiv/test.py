# Test PyTorch NCCL
import torch
import torch.distributed as dist
torch.cuda.current_device()
torch.cuda.get_device_name()



dist.init_process_group(backend="nccl")
local_rank = dist.get_rank() % torch.cuda.device_count()
import os

dist_vars = [
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NCCL_SOCKET_IFNAME",
    "NCCL_DEBUG",
    "CUDA_VISIBLE_DEVICES",
]

print("\n--- torch.distributed env vars ---")
for var in dist_vars:
    print(f"{var}: {os.environ.get(var)}")


print('Hallo1')
torch.cuda.set_device(local_rank)
print('Hallo2')
data = torch.FloatTensor([1,] * 128).to("cuda")
print('Hallo3')
dist.all_reduce(data, op=dist.ReduceOp.SUM)
print('Hallo4')
torch.cuda.synchronize()
print('Hallo5')
value = data.mean().item()
print('Hallo6')
world_size = dist.get_world_size()
print('Hallo7')
print(value)
print(world_size)
assert value == world_size, f"Expected {world_size}, got {value}"

print("PyTorch NCCL is successful!")
print('Hallo21')
print('Hallo21')
print('Hallo21')
print('Hallo21')
print('Hallo21')
# Test PyTorch GLOO
gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
cpu_data = torch.FloatTensor([1,] * 128)
dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
value = cpu_data.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("PyTorch GLOO is successful!")

if world_size <= 1:
    exit()

# Test vLLM NCCL, with cuda graph
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
# pynccl is enabled by default for 0.6.5+,
# but for 0.6.4 and below, we need to enable it manually.
# keep the code for backward compatibility when because people
# prefer to read the latest documentation.
pynccl.disabled = False

s = torch.cuda.Stream()
with torch.cuda.stream(s):
    data.fill_(1)
    out = pynccl.all_reduce(data, stream=s)
    value = out.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"

print("vLLM NCCL is successful!")

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(cuda_graph=g, stream=s):
    out = pynccl.all_reduce(data, stream=torch.cuda.current_stream())

data.fill_(1)
g.replay()
torch.cuda.current_stream().synchronize()
value = out.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("vLLM NCCL with cuda graph is successful!")

dist.destroy_process_group(gloo_group)
dist.destroy_process_group()
