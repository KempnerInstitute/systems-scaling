import os
import sys
import copy
import logging
from pathlib import Path
from typing import Optional, TextIO
from packaging import version
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
from torch.distributed.fsdp.wrap import wrap

from olmo.config import CheckpointType, TrainConfig
from olmo.data import (
    build_train_dataloader,
    build_sft_dataloader,
    build_train_dataloader_fixed_index,
)
from olmo.eval import build_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import BoltOnWarmupScheduler, build_optimizer, build_scheduler
from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    peak_gpu_memory,
    seed_all,
)
from olmo.train import Trainer
from olmo.util import clean_opt, log_extra_field, prepare_cli_environment
from olmo.registry import MODEL_DICT, INDEX_DICT

from mx import finalize_mx_specs, mx_mapping

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

log = logging.getLogger("train")
def build_models(cfg: TrainConfig):
    # Initialize the model.
    log.info("Building model...")
    # MXFP8_e5m2 matmuls with bfloat16 vector ops, forward pass only
    mx_specs = {
            'scale_bits': 8,
            'w_elem_format': cfg.model.w_mx_format,
            'a_elem_format': cfg.model.a_mx_format,
            'block_size': 32,
            'bfloat': 16,
            'custom_cuda': True,
            # For quantization-aware finetuning, do backward pass in FP32
            'quantize_backprop': True,
        }
    mx_specs = finalize_mx_specs(mx_specs)
    mx_mapping.inject_pyt_ops(mx_specs)

    olmo_model = OLMo(cfg.model)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    log.info(f"Peak GPU Memory (MB) before FSDP: {int(peak_gpu_memory() or 0)}")

    olmo_model.set_activation_checkpointing(cfg.activation_checkpointing)

    # Wrap the model in FSDP.
    log.info("Wrapping model with FDSP...")
    wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)

    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        # This prevents any parameters from being initialized twice
        def dummy_init_fn(module: torch.nn.Module) -> None:
            module.to_empty(device=get_default_device())

        param_init_fn = dummy_init_fn
    else:
        param_init_fn = None

    # Set up device mesh for hybrid sharding in order to specify which nodes are assoicated to a given model replica
    device_mesh = None
    hybrid_sharding_fsdp_kwargs = {}
    if cfg.fsdp.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        if version.parse(torch.__version__) < version.parse("2.2.0"):
            # Device mesh was not added to PyTorch until v2.2.0
            raise OLMoConfigurationError(
                "OLMo training does not correctly support hybrid sharding before torch 2.2.0"
            )

        from torch.distributed.device_mesh import init_device_mesh

        num_model_replicas = cfg.fsdp.hybrid_sharding_num_model_replicas or (
            get_world_size() // get_local_world_size()
        )

        if num_model_replicas <= 0:
            raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must be a positive integer")

        num_nodes = get_world_size() // get_local_world_size()
        if num_nodes > 1 and num_nodes % num_model_replicas != 0:
            raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must divide number of nodes")

        device_mesh = init_device_mesh("cuda", (num_model_replicas, get_world_size() // num_model_replicas))
        hybrid_sharding_fsdp_kwargs["device_mesh"] = device_mesh

    fsdp_model = FSDP(
        olmo_model,
        sharding_strategy=cfg.fsdp.sharding_strategy,
        mixed_precision=cfg.fsdp_precision,
        auto_wrap_policy=wrap_policy,
        use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
        limit_all_gathers=True,
        device_id=get_local_rank(),
        param_init_fn=param_init_fn,
        **hybrid_sharding_fsdp_kwargs,
    )
    # when param_init_fn is None, FSDP will call reset_parameters() automatically
    if param_init_fn is not None:
        olmo_model.reset_parameters()

    log.info(f"Peak GPU Memory (MB) after FSDP: {int(peak_gpu_memory() or 0)}")
    log.info("Model:")
    log.info(fsdp_model)
    return olmo_model, fsdp_model

def train(rank, world_size, cfg):
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

    for epoch in range(1):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

    # Ensure run name set.
    if cfg.run_name is None:
        raise OLMoConfigurationError("--run_name is required")
    log_extra_field("run_name", cfg.run_name)

    # Check for none path
    if cfg.load_path == "None":
        cfg.load_path = None

    # Check for requeue ckpt
    rq_path = Path(cfg.save_folder) / "latest"
    if rq_path.exists() and cfg.requeue:
        cfg.load_path = str(rq_path)
        log.info(f"Requeueing from {cfg.load_path}")

    # Sanity check
    if (cfg.reset_optimizer_state or cfg.reset_trainer_state) and cfg.load_path is None:
        log.warning(
            "You want to reset the optimizer or trainer state, but we're not loading from the checkpoint. The"
            "setting has no effect."
        )

    barrier()

    # Set CUDA device.
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.training.batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    if cfg.optimizer.no_decay_norm_and_bias is not None:
        log.warning(
            "You set the deprecated config option `no_decay_norm_and_bias`. For compatibility, this"
            "setting will take precedence over all other weight decay configurations. Please change"
            "your config to use `decay_norm_and_bias` and `decay_embeddings` instead."
        )
        cfg.optimizer.decay_norm_and_bias = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.decay_embeddings = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.no_decay_norm_and_bias = None  # So nobody uses this by accident.

    # Display and save configuration.
    if get_global_rank() == 0:
        if cfg.datasets.paths is not None and len(cfg.datasets.paths) < 50:
            log.info("Configuration:")
            log.info(cfg)
        if not cfg.dry_run and (cfg.load_path is None or Path(cfg.load_path).parent != Path(cfg.save_folder)):
            # Save config.
            save_path = Path(cfg.save_folder) / "config.yaml"
            if save_path.is_file() and not cfg.save_overwrite:
                raise OLMoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")
            else:
                log.info(f"Saving config to {save_path}")
                save_path.parent.mkdir(exist_ok=True, parents=True)
                cfg.save(save_path)
            del save_path

    barrier()

    # Maybe start W&B run.
    if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=wandb_dir,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=cfg.asdict(exclude=["wandb"]),
        )

    barrier()

    # Set seed.
    seed_all(cfg.seed)

    # Construct data loader
    if cfg.datasets.index_path is not None:
        log.info(f"Using fixed index path: {cfg.datasets.index_path}")
        cfg.datasets.index_path = INDEX_DICT.get(cfg.datasets.index_path, cfg.datasets.index_path)
        train_loader = build_train_dataloader_fixed_index(cfg)
    elif cfg.sft_dataset is not None:
        dataset_labels = cfg.sft_dataset.label.split(",")
        data_cfgs = []
        for label in dataset_labels:
            data_cfg = copy.deepcopy(cfg.sft_dataset)
            data_cfg.label = label
            data_cfgs.append(data_cfg)
        train_loader = build_sft_dataloader(cfg, data_cfgs)
    else:
        train_loader = build_train_dataloader(cfg)

    # train_loader, val_loader = data.create_dataloaders(cfg)
    # log.info(f"Built train dataloader for dataset of size {train_loader.dataset.total_size}")


if __name__ == "__main__":
    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")
    world_size = 4 # torch.cuda.device_count()

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])

    mp.spawn(train, args=(world_size, cfg), nprocs=world_size, join=True)