import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    # Initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=size,
        rank=rank
    )
    
    # Create a large tensor (~400MB)
    tensor_size = 1024 * 1024 * 100  # 100 million elements
    if rank == 0:
        tensor = torch.randn(tensor_size, dtype=torch.float32).cuda()
    else:
        tensor = torch.zeros(tensor_size, dtype=torch.float32).cuda()
    
    # Warm-up broadcast
    dist.broadcast(tensor, src=0)
    dist.barrier()
    
    # Ensure all CUDA operations are completed before timing
    torch.cuda.synchronize()
    dist.barrier()
    
    # Measure the time for multiple broadcasts
    num_iters = 10
    start_time = time.time()
    for _ in range(num_iters):
        dist.broadcast(tensor, src=0)
    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()
    
    # Calculate average time and bandwidth
    avg_time = (end_time - start_time) / num_iters
    data_size_bytes = tensor.element_size() * tensor.nelement()
    data_size_gb = data_size_bytes / 1e9
    bandwidth = data_size_gb / avg_time  # in GB/s
    
    if rank == 0:
        print(f"Average broadcast time: {avg_time:.6f} seconds")
        print(f"Data size per broadcast: {data_size_gb:.2f} GB")
        print(f"Effective bandwidth: {bandwidth:.2f} GB/s")
    
    # Clean up
    dist.destroy_process_group()

def main():
    size = 2  # Number of GPUs/processes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(run, nprocs=size, args=(size,))
    
if __name__ == '__main__':
    main()
