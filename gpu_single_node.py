import torch
import itertools

def simulate_matmul_with_sharding():
    # Number of GPUs
    num_gpus = 4
    assert torch.cuda.device_count() >= num_gpus, "Simulation requires at least 4 GPUs."

    # Dimensions of the matrix
    shard_dim = 512  # Dimension of each shard
    total_dim = shard_dim * 2

    # Generate random matrices for weights and inputs
    W = torch.randn(total_dim, total_dim, device="cpu")
    X = torch.randn(total_dim, total_dim, device="cpu")

    # Shard weights and inputs
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
    required_weight_shards = [
        [0, 1],
        [0, 1],
        [2, 3],
        [2, 3]
    ]
    required_input_shards = [
        [0, 1],
        [2, 3],
        [0, 1],
        [2, 3]
    ]

    # Fixed weight-to-GPU assignments
    weight_to_gpu = {gpu: weight_shards[gpu].to(f"cuda:{gpu}") for gpu in range(4)}

    # Generate all permutations of input shard assignments to GPUs
    input_orders = list(itertools.permutations(range(num_gpus)))

    # Track results
    results = []

    # Simulate all input-to-GPU assignments
    for order in input_orders:
        # Log which GPU holds which input shard
        input_to_gpu = {gpu: input_shards[idx].to(f"cuda:{gpu}") for idx, gpu in enumerate(order)}

        # Simulate matrix multiplication
        communication_log = []
        result_shards = {}
        comm_times = []

        for gpu_id in range(num_gpus):
            # Each GPU computes one quadrant of the final result
            torch.cuda.set_device(gpu_id)
            local_result = torch.zeros(shard_dim, shard_dim, device=f"cuda:{gpu_id}")

            # Collect weights and inputs needed for the computation
            for weight_shard_id, input_shard_id in zip(required_weight_shards[gpu_id], required_input_shards[gpu_id]):
                
                weight_shard = weight_to_gpu[weight_shard_id] #.to(f"cuda:{gpu_id}")
                input_shard = input_to_gpu[input_shard_id] #.to(f"cuda:{gpu_id}")

                # Simulate shard transfer if input shard is not local
                print('gpu_id', gpu_id, 'weight_shard_id', weight_shard_id, input_shard_id)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                if weight_shard.device.index != gpu_id:
                    start_event.record()  # Record start of weight shard transfer
                    communication_log.append((f"W_{weight_shard.device.index}", weight_shard.device, gpu_id))
                    weight_shard = weight_shard.clone().to(f"cuda:{gpu_id}")
                    end_event.record()  # Record end of weight shard transfer
                    torch.cuda.synchronize()  # Synchronize to ensure the transfer is complete
                    comm_times.append(start_event.elapsed_time(end_event))  # Measure time for transfer

                if input_shard.device.index != gpu_id:
                    start_event.record()  # Record start of input shard transfer
                    communication_log.append((f"x_{input_shard_id}", input_shard.device, gpu_id))
                    input_shard = input_shard.clone().to(f"cuda:{gpu_id}")
                    end_event.record()  # Record end of input shard transfer
                    torch.cuda.synchronize()  # Synchronize to ensure the transfer is complete
                    comm_times.append(start_event.elapsed_time(end_event))  # Measure time for transfer
                
                # Compute partial results for the corresponding submatrix
                local_result += weight_shard @ input_shard.T

            result_shards[gpu_id] = local_result

        # Store results
        results.append({
            "input_order": order,
            "communication_log": communication_log,
            'comm_cost': sum(comm_times)
        })

    # Display results
    for res in results:
        print(f"Input Order: {res['input_order']}")
        print(f"Communication Log: {res['communication_log']}")
        print(f"Communication cost (in milliseconds): {res['comm_cost']}")
        print("-" * 50)

if __name__ == "__main__":
    simulate_matmul_with_sharding()