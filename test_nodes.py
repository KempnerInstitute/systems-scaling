import torch
import torch.distributed as dist
import os

def main():
    # Read rank and world size directly from environment variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    x = torch.tensor([rank], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    print(f"Rank {rank}/{world_size} on node {os.environ['SLURMD_NODENAME']} - sum = {x.item()}")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()