import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
from torch.distributed.fsdp.wrap import wrap

def setup(rank, world_size):
    """Initialize the process group for FSDP."""
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", init_method=f"tcp://{master_addr}:{master_port}", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Destroy the process group after training."""
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    """A basic model for FSDP testing."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(rank, world_size):
    """Training loop for each process in FSDP."""
    setup(rank, world_size)

    torch.manual_seed(42)
    model = SimpleModel().to(rank)

    # Wrap model in FSDP
    model = FSDP(model, 
                 cpu_offload=CPUOffload(offload_params=False), 
                 backward_prefetch=BackwardPrefetch.BACKWARD_PRE)

    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Dummy input and target
    x = torch.randn(16, 10).to(rank)
    y = torch.randn(16, 1).to(rank)

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    test_forward_backward(rank, model, criterion)

    cleanup()

def test_forward_backward(rank, model, criterion):
    """Test forward and backward pass for correctness."""
    model.eval()  # Disable dropout, batchnorm updates

    with torch.no_grad():
        test_input = torch.randn(8, 10).to(rank)
        test_output = model(test_input)
        print(f"Rank {rank} Test Forward Output: {test_output}")

    model.train()

    test_input = torch.randn(8, 10).to(rank)
    test_target = torch.randn(8, 1).to(rank)

    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    output = model(test_input)
    loss = criterion(output, test_target)
    loss.backward()

    # Print gradient norms to verify updates
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Rank {rank}, {name} Grad Norm: {param.grad.norm().item()}")

    optimizer.step()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)