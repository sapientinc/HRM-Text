import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

x = torch.ones(1024, 1024, device="cuda") * (rank + 1)
dist.all_reduce(x)
expected = sum(r + 1 for r in range(world_size))
ok = torch.allclose(x, torch.full_like(x, float(expected)))
print(f"[rank {rank}/{world_size}] all_reduce result matches expected ({expected}): {ok}", flush=True)

dist.barrier()
if rank == 0:
    print("NCCL 2-node test PASSED", flush=True)
dist.destroy_process_group()
