import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import itertools
import time
import argparse

def log_gpu_memory(rank, message, gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.synchronize(device=device)
    allocated = torch.cuda.memory_allocated(device=device)
    reserved = torch.cuda.memory_reserved(device=device)
    max_allocated = torch.cuda.max_memory_allocated(device=device)
    max_reserved = torch.cuda.max_memory_reserved(device=device)

    print(f"Rank {rank} - {message}")
    print(f"Allocated Memory: {allocated / (1024 ** 2):.2f} MB")
    print(f"Reserved Memory: {reserved / (1024 ** 2):.2f} MB")
    print(f"Max Allocated Memory: {max_allocated / (1024 ** 2):.2f} MB")
    print(f"Max Reserved Memory: {max_reserved / (1024 ** 2):.2f} MB")
    print("-" * 40)


def gather_communication_times(communication_times, rank, world_size):
    if rank == 0:
        # Prepare a list to gather communication times from all processes
        gather_list = [torch.zeros_like(communication_times) for _ in range(world_size)]
    else:
        gather_list = None

    dist.gather(communication_times, gather_list, dst=0)

    if rank == 0:
        # Process the gathered communication times
        print(f" Communication Times:")
        inter_node_times = 0
        intra_node_times = 0
        total_comm_times = 0
        for idx, comm_times in enumerate(gather_list):
            inter_node_time = comm_times[0].item()
            intra_node_time = comm_times[1].item()
            inter_node_times += inter_node_time
            intra_node_times += intra_node_time
            total_comm_times += inter_node_time + intra_node_time
            
        print(f"  Inter-node Time: {inter_node_times}")
        print(f"  Intra-node Time: {intra_node_times}")
        print(f"  Total Communication Time: {total_comm_times}")
        print("-" * 50)


def get_required_weight_shards(rank):
    return [0, 1] if rank < 2 else [2, 3]


def get_required_input_shards(rank, order):
    if rank % 2 == 0:
        return [order[0], order[1]]
    else:
        return [order[2], order[3]]

def run_process(shard_dim, weight_shards, input_shards):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    
    num_gpus_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    node_id = rank // num_gpus_per_node
    gpu_id = rank % num_gpus_per_node
    torch.cuda.set_device(gpu_id)

    # Initialize communication time trackers
    inter_node_time = 0
    intra_node_time = 0

    # Map shard IDs to process ranks
    weight_shard_id_to_rank = {
        0: 0,
        1: 1,
        2: 2,
        3: 3
    }
    order = [int(i) for i in os.environ['ORDER'].split(',')]
    input_shard_id_to_rank = {input_shard_id: rank for rank, input_shard_id in enumerate(order)}

    # Determine which shards this process requires
    

    required_weight_shards = get_required_weight_shards(rank)
    required_input_shards = get_required_input_shards(rank, order)

    received_weight_shards = {}
    received_input_shards = {}

    # Allocate space for local result
    local_result = torch.zeros(shard_dim, shard_dim, device=f"cuda:{gpu_id}")

    world_size = 4

    # Broadcast input shards
    input_groups = []
    for input_shard_id in range(world_size):

        input_owner_rank = input_shard_id_to_rank[input_shard_id]

        ranks_needing_input_shard = [
            r for r in range(world_size) if input_shard_id in get_required_input_shards(r, order)
        ]
        # print(input_shard_id, 'p1', ranks_needing_input_shard)
        
        if input_owner_rank not in ranks_needing_input_shard:
            ranks_needing_input_shard.append(input_owner_rank)

        ranks_needing_input_shard.sort()

        group = dist.new_group(ranks=ranks_needing_input_shard)
        input_groups.append((group, input_shard_id, input_owner_rank, ranks_needing_input_shard))

    print('input_groups', input_groups)
    for group_info in input_groups:
        
        group, input_shard_id, input_owner_rank, group_ranks = group_info
        
        in_group = rank in group_ranks
        if in_group:
            if rank == input_owner_rank:
                input_shard = input_shards[input_shard_id].to(gpu_id)
            else:
                input_shard = torch.zeros(shard_dim, shard_dim, device=f"cuda:{gpu_id}")

            torch.cuda.synchronize()
            dist.barrier(group=group)

            start_time = time.perf_counter()

            dist.broadcast(tensor=input_shard, src=input_owner_rank, group=group)

            communication_time = time.perf_counter() - start_time
            print('input', communication_time)


            # Update communication time if this process needs the shard
            if input_shard_id in required_input_shards:
                if node_id != input_owner_rank // num_gpus_per_node:
                    inter_node_time += communication_time
                else:
                    intra_node_time += communication_time
                received_input_shards[input_shard_id] = input_shard

        else:
            pass

    # Broadcast weight shards
    weight_groups = []
    for weight_shard_id in range(world_size):
        weight_owner_rank = weight_shard_id_to_rank[weight_shard_id]
        ranks_needing_weight_shard = [
            r for r in range(world_size) if weight_shard_id in get_required_weight_shards(r)
        ]
        if weight_owner_rank not in ranks_needing_weight_shard:
            ranks_needing_weight_shard.append(weight_owner_rank)
        ranks_needing_weight_shard.sort()
        group = dist.new_group(ranks=ranks_needing_weight_shard)
        weight_groups.append((group, weight_shard_id, weight_owner_rank, ranks_needing_weight_shard))
    
    for group_info in weight_groups:
        group, weight_shard_id, weight_owner_rank, group_ranks = group_info
        in_group = rank in group_ranks
        if in_group:
            if rank == weight_owner_rank:
                weight_shard = weight_shards[weight_shard_id].to(gpu_id)
            else:
                weight_shard = torch.zeros(shard_dim, shard_dim, device=f"cuda:{gpu_id}")

            torch.cuda.synchronize()
            dist.barrier(group=group)

            start_time = time.perf_counter()

            dist.broadcast(tensor=weight_shard, src=weight_owner_rank, group=group)

            communication_time = time.perf_counter() - start_time
            print('weight', communication_time)

            # Update communication time if this process needs the shard
            if weight_shard_id in required_weight_shards:
                if node_id != weight_owner_rank // num_gpus_per_node:
                    inter_node_time += communication_time
                else:
                    intra_node_time += communication_time
                received_weight_shards[weight_shard_id] = weight_shard


    # Perform computation
    for weight_shard_id in required_weight_shards:
        weight_shard = received_weight_shards[weight_shard_id]
        for input_shard_id in required_input_shards:
            input_shard = received_input_shards[input_shard_id]
            local_result += weight_shard @ input_shard.T

    # Log results for this process
    communication_times = torch.tensor(
        [inter_node_time, intra_node_time], dtype=torch.float32, device=f"cuda:{gpu_id}"
    )
    gather_communication_times(communication_times, rank, world_size)


    # Finalize process group
    dist.barrier()
    dist.destroy_process_group()

    


# Main function to spawn processes
def simulate_matmul_with_distributed_sharding():
    # Number of GPUs and nodes
    num_nodes = 2
    num_gpus_per_node = 2
    world_size = num_nodes * num_gpus_per_node
    shard_dim = 4096  # Dimension of each shard

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

    # run_process,
    #     args=(world_size, num_gpus_per_node, results, shard_dim, weight_shards, input_shards),
    #     nprocs=world_size,
    #     join=True,
    # os.environ['ORDER'] = ','.join(map(str, order))
    run_process(shard_dim=shard_dim, weight_shards=weight_shards, input_shards=input_shards)



if __name__ == "__main__":
    # input_orders = list(itertools.permutations(range(4)))
    # print(input_orders)
    # for order in input_orders:
    #     print('order', order)

    parser = argparse.ArgumentParser()
    parser.add_argument('--order', type=str, required=True, help='Comma-separated order indices')
    args = parser.parse_args()
    os.environ['ORDER'] = args.order
    simulate_matmul_with_distributed_sharding()

    #     torch.cuda.empty_cache()
    
# if __name__ == "__main__":
#     init_process()