import yaml
import numpy as np
from olmo.model import OLMo
from olmo.config import ModelConfig

train_sets = [
    "fineweb-100b",
    "starcoder",
    "proof-pile-2",
    "fineweb-edu-100b",
    "slimpajama-chunk1",
    "smollm-corpus",
]

sweep_config = {
    "wandb": {"group": "new-scale-fixed-tokens"},
    "sweep": [],
    "save_num_checkpoints_to_keep": 1,
    "save_num_unsharded_checkpoints_to_keep": 1,
    "save_interval": 1000,
    "save_interval_unsharded": 100000,
    "data": {"paths": train_sets},
    "global_train_batch_size": 512,
}

fixed_tokens = int(1.3e11)  # at least 100B tokens

model_sizes = [
    (704, 11),
    (832, 13),
    (960, 15),
    (1024, 16),  # 260M
    (1152, 18),
    (1216, 19),  # 450M
    (1344, 21),
    (1408, 22),  # 610M
    (1664, 26),
    (1920, 30),
]

vocab_size = 32000

model_defaults = {
    "context_length": 512,
    "mlp_ratio": 4,
    "rope": True,
    "attention_layer_norm": True,
    "attention_layer_norm_with_affine": True,
    "multi_query_attention": False,
    "include_bias": False,
    "block_type": "sequential",
    "layer_norm_type": "default",
    "layer_norm_with_affine": True,
    "bias_for_layer_norm": False,
    "activation_type": "gelu",
    "attention_dropout": 0.0,
    "residual_dropout": 0.0,
    "embedding_dropout": 0.0,
    "vocab_size": vocab_size,
    "embedding_size": vocab_size,
    "eos_token_id": 1,
    "pad_token_id": 0,
    "init_device": "meta",
    "init_fn": "mitchell",
    "weight_tying": False,
}


def get_model_config_and_size(d_model, n_layers):
    head_size = 64
    model_config = ModelConfig(**model_defaults)
    model_config.d_model = d_model
    model_config.n_layers = n_layers
    model_config.n_heads = d_model // head_size

    model = OLMo(model_config, False)
    params = model.num_params()
    fwd_flops = model.num_fwd_flops

    seq_len = model_defaults["context_length"]
    activation_memory = 2 * (n_layers + 1) * (seq_len * d_model) * (4 + 4 + 2)
    attention_memory = 2 * n_layers * seq_len**2 * d_model // head_size
    output_memory = 2 * 4 * seq_len * vocab_size
    model_memory = 3 * 4 * params

    bs = int((125e9 - model_memory) / (activation_memory + attention_memory + output_memory)) # CHANGED TO H200
    bs = min(2 ** int(np.log2(bs)), 256) # round down to the nearest power of 2, max 256 (changed from 128)

    return (
        params,
        fwd_flops,
        {
            "model": {
                "d_model": d_model,
                "n_heads": d_model // head_size,
                "n_layers": n_layers,
                "context_length": seq_len,
            },
            "device_train_microbatch_size": bs,
            "device_eval_batch_size": min(bs, 64),
        },
    )


def expand_config(params, fwd_flops, config, tokens=fixed_tokens, global_bs=sweep_config["global_train_batch_size"]):
    tok_bs = global_bs * config["model"]["context_length"]
    steps = int(tokens / tok_bs)
    total_flops = steps * tok_bs * 6 * params

    config.update({
        "total_flops": total_flops,
        "max_duration": steps,
        "scheduler": {"t_warmup": steps // 5},
        "eval_interval": steps // 10,
        "params": params,
        "tokens": tokens,
        "ratio": tokens / params,
    })

    print(f"Model size: {params}, Tokens: {tokens}, Steps: {steps}, FLOPs: {total_flops}")

    return config


if __name__ == "__main__":
    for d, l in model_sizes:
        params, fwd_flops, config = get_model_config_and_size(d, l)
        expanded_config = expand_config(params, fwd_flops, config)

        sweep_config["sweep"].append(expanded_config)

    print(f"Total sweep size: {len(sweep_config['sweep']) * len(train_sets)}")
    print(f"Fixed tokens per run: {fixed_tokens}")

    with open("configs/sweeps/scale_fixed_tokens_overtrained_new.yaml", "w") as f:
        yaml.dump(sweep_config, f)