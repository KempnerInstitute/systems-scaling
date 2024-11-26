import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import itertools
import time

# Function for each GPU process
def run_process(rank, world_size, num_gpus_per_node, results, shard_dim, weight_shards, input_shards):
    # os.environ["MASTER_ADDR"] = "localhost"  # Replace with master node IP if running on multiple nodes
    # os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Rank: {torch.distributed.get_rank()}, GPU: {torch.cuda.current_device()}")

    # Node and GPU assignments
    node_id = rank // num_gpus_per_node
    gpu_id = rank % num_gpus_per_node
    torch.cuda.set_device(gpu_id)

    # Track communication
    communication_cost = 0

    # Required shards for the process's computation
    required_weight_shards = [0, 1] if gpu_id < 2 else [2, 3]
    required_input_shards = [0, 1] if gpu_id % 2 == 0 else [2, 3]

    # Allocate space for local result
    local_result = torch.zeros(shard_dim, shard_dim, device=f"cuda:{gpu_id}")

    for weight_shard_id, input_shard_id in zip(required_weight_shards, required_input_shards):
        # Transfer weight shards if needed
        weight_shard = weight_shards[weight_shard_id]
        start_time = time.perf_counter()
        if node_id != weight_shard_id // num_gpus_per_node:
            # Inter-node communication
            dist.send(tensor=weight_shard, dst=weight_shard_id)
            dist.recv(tensor=weight_shard, src=weight_shard_id)
            inter_node_time += time.perf_counter() - start_time
        elif gpu_id != weight_shard_id:
            # Intra-node communication
            dist.send(tensor=weight_shard, dst=weight_shard_id)
            dist.recv(tensor=weight_shard, src=weight_shard_id)
            intra_node_time += time.perf_counter() - start_time

        # Measure input shard communication
        input_shard = input_shards[input_shard_id]
        start_time = time.perf_counter()
        if node_id != input_shard_id // num_gpus_per_node:
            # Inter-node communication
            dist.send(tensor=input_shard, dst=input_shard_id)
            dist.recv(tensor=input_shard, src=input_shard_id)
            inter_node_time += time.perf_counter() - start_time
        elif gpu_id != input_shard_id:
            # Intra-node communication
            dist.send(tensor=input_shard, dst=input_shard_id)
            dist.recv(tensor=input_shard, src=input_shard_id)
            intra_node_time += time.perf_counter() - start_time

        # Compute partial results
        local_result += weight_shard @ input_shard.T

    # Log results for this process
    results[gpu_id] = {
        "rank": rank,
        "local_result": local_result,
        "communication_cost": communication_cost,
    }

    # Finalize process group
    dist.barrier()
    dist.destroy_process_group()

# Main function to spawn processes
def simulate_matmul_with_distributed_sharding(order):
    # Number of GPUs and nodes
    num_nodes = 2
    num_gpus_per_node = 2
    world_size = num_nodes * num_gpus_per_node
    shard_dim = 512  # Dimension of each shard

    # Dimensions
    total_dim = shard_dim * 2

    # Simulate random shards for weights and inputs
    W = torch.randn(total_dim, total_dim, device="cpu")
    X = torch.randn(total_dim, total_dim, device="cpu")

    weight_shards = [
        W[:shard_dim, :shard_dim],  # W_11
        W[:shard_dim, shard_dim:],  # W_12
        W[shard_dim:, :shard_dim],  # W_21
        W[shard_dim:, shard_dim:],  # W_22
    ]
    input_shards = [
        X[:shard_dim, :shard_dim],  # x_11
        X[:shard_dim, shard_dim:],  # x_12
        X[shard_dim:, :shard_dim],  # x_21
        X[shard_dim:, shard_dim:],  # x_22
    ]

    # Move local weight shards to GPU
    # input_shards = [shard.to(f"cuda:{gpu_id}") for shard in input_shards]
    # input_to_gpu = {gpu: input_shards[idx].to(f"cuda:{gpu}") for idx, gpu in enumerate(order)}

    manager = mp.Manager()
    results = manager.dict()
    # run_process,
    #     args=(world_size, num_gpus_per_node, results, shard_dim, weight_shards, input_shards),
    #     nprocs=world_size,
    #     join=True,
    run_process(rank=int(os.environ["RANK"]), world_size=world_size, 
                num_gpus_per_node=num_gpus_per_node, results=results, 
                shard_dim=shard_dim, weight_shards=weight_shards, input_shards=input_shards)


    # Display results
    for gpu_id, res in results.items():
        print(f"GPU {gpu_id} (Rank {res['rank']}):")
        print(f"Local Result:\n{res['local_result']}")
        print(f"Communication Cost: {res['communication_cost']}")
        print("-" * 50)

if __name__ == "__main__":
    input_orders = list(itertools.permutations(range(4)))
    for order in input_orders:
        print(order)
        simulate_matmul_with_distributed_sharding(order)