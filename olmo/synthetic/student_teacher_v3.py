import argparse, pathlib, random, csv, time, yaml, math, warnings
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass, field


import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import wandb
from torch.autograd import Function
from torch import Tensor

import copy

STORE_GRADS_EVERY = 5
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def gaussian(batch, dim, device="cpu"): return torch.randn(batch, dim, device=device)

# mx_utils.py
FORMAT_META = {
    "fp8_e4m3": dict(emax=8,   max_normal=448.0),   # 1111 110₂ × 2⁸ × 1.75
    "fp8_e5m2": dict(emax=15,  max_normal=57344.0),
    "fp6_e3m2": dict(emax=7,   max_normal=224.0),
    "fp4_e2m1": dict(emax=3,   max_normal=6.0),
    "bfloat16": dict(emax=127, max_normal=65504.0),  # 1111 1111₂ × 2¹² × 1.0
}

def count_clipped_values(t: torch.Tensor,
                         block_size: int,
                         fmt: str
) -> tuple[int, int]:
    """
    Return (#clipped, #total) for a tensor that would flow into the lagest buckets in
    block-wise quantization with format `fmt`.
    """
    emax, vmax = fmt_meta(fmt)           # vmax = max representable after scaling
    flat = t.detach().abs().view(-1, block_size)   #   (n_blocks, block_size)
    max_per_block = flat.max(dim=1, keepdim=True).values
    scales = 2.0 ** (torch.floor(torch.log2(max_per_block)) - emax)
    clipped = (flat > vmax * scales).sum().item()
    total   = flat.numel()
    return clipped, total

def count_clipped_values_fp32(
    t: torch.Tensor,
    block_size: int = None,   # unused for fp32
    fmt: str = "fp32"         # unused
) -> tuple[int,int]:
    """
    Return (#clipped, #total) for a tensor in FP32,
    i.e. outside ±FLT_MAX.
    """
    max_f32 = torch.finfo(torch.float32).max
    clipped = (t.abs() > max_f32).sum().item()
    total   = t.numel()
    return clipped, total

def fmt_meta(fmt):
    if fmt not in FORMAT_META:
        raise ValueError(f"unknown MX format {fmt}")
    return FORMAT_META[fmt]["emax"], FORMAT_META[fmt]["max_normal"]


def grad_vec(model, dtype=None, *, drop_scalars=False):
    if dtype is None:
        dtype = next(model.parameters()).dtype
    pieces = []
    for name, p in model.named_parameters():
        if drop_scalars and (name.endswith(".g") or name.endswith(".g_a1") or name.endswith(".g_a2")):
            continue
        if p.grad is not None:
            pieces.append(p.grad.flatten().to(dtype))
        else:
            pieces.append(torch.zeros_like(p, dtype=dtype).flatten())
    return torch.cat(pieces)

def save_eps(run_dir, step, eps, group_dict):
    p = run_dir / "eps"; p.mkdir(exist_ok=True)
    torch.save({"step": step,
                "eps":  eps,
                "group_var": group_dict},
               p / f"eps_step{step:04d}.pt")

def val_loss(model, teacher, crit, val_batches, device=None):
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        losses = [crit(model(v.to(device)), teacher(v.to(device))) for v in val_batches]
    return torch.stack(losses).mean().item()

# ===================  MODEL  =====================================
def swiglu(x):
    a,b = x.chunk(2, -1)
    return a * F.silu(b)

def swiglu(a, b):
    return a * F.silu(b)


# ================ UNUSED DEBIASING CODE =========================
# the below debiased networks try to implement a layerwise learning rate
# but they do not work super well in practice (or at least, not really better
# than just lowering the overall learning rate)... this is unused in our
# paper but we keep it here for reference.

class DebiasGate(Function):
    @staticmethod
    def forward(ctx, w: Tensor, g: Tensor) -> Tensor:
        ctx.save_for_backward(w, g)
        return w                    # identity forward

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        w, g = ctx.saved_tensors
        # gradient wrt weight
        grad_w = grad_out * g
        # gradient wrt scalar g  =  ⟨grad_out , w⟩
        grad_g = (grad_out * w).sum().unsqueeze(0)
        return grad_w, grad_g

import math
LN2        = math.log(2.0)
G_W_INIT   = LN2 ** 3
G_B_INIT   = LN2 ** 2

class LinearDebiased(nn.Linear):
    """
    nn.Linear whose weight gradient is multiplied by a learnable scalar g.
    Initialise g to (ln 2)**3 ≈ 0.333 for MX-quantised W⊗X layers.
    """
    def __init__(self, in_f, out_f, bias=True, init_g=(math.log(2)**3)):
        super().__init__(in_f, out_f, bias=bias)
        self.g = nn.Parameter(torch.tensor([init_g], dtype=self.weight.dtype))
        

    def forward(self, x):
        w_eff = DebiasGate.apply(self.weight, self.g)
        return F.linear(x, w_eff, self.bias)

class ActDebiasGate(Function):
    """
    Pass-through in the forward; scales the *incoming*
    gradient by the learnable scalar g_a in the backward.
    """
    @staticmethod
    def forward(ctx, x: Tensor, g_a: Tensor) -> Tensor:
        ctx.save_for_backward(g_a)
        return x

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (g_a,) = ctx.saved_tensors
        return grad_out * g_a, None           # grad wrt x is scaled; no grad for g_a

def debias_act(x: Tensor, g_a: nn.Parameter) -> Tensor:
    # identical value, corrected gradient
    return ActDebiasGate.apply(x, g_a)

# =======================================================

class ResidualBlock(nn.Module):
    def __init__(self, d, ln: bool = True, act: str = "gelu", debias: bool = False):
        super().__init__()
        self.ln = nn.LayerNorm(d) if ln else nn.Identity()
        act = act.lower()
        if act not in {"gelu", "relu", "linear", "swiglu",}:
            raise ValueError(f"Unsupported activation: {act}")
        self.act = act
        self.debias = debias

        if debias:
            self.g_a1 = nn.Parameter(torch.tensor([math.log(2.0)]))
            self.g_a2 = nn.Parameter(torch.tensor([math.log(2.0)]))

            if act == "swiglu":
                # keep 8 d² params  ⇒  h ≈ 8/3 ⋅ d
                h  = int(round(8 * d / 3))
                self.fc1 = LinearDebiased(d, 2 * h, bias=False, init_g=G_W_INIT)
                self.fc2 = LinearDebiased(h,     d, bias=True,  init_g=G_W_INIT)
                # bias needs its *own* scalar (2 scales) – register a hook
                self.fc2.bias.register_hook(lambda grad: grad * G_B_INIT)
                self.act_fn = lambda u: swiglu(*u.chunk(2, dim=-1))

                #self.fc2.bias.register_hook(lambda g: g.mul_(G_B_INIT))
                self.act_fn = lambda u: swiglu(*u.chunk(2, dim=-1))
            elif (act == "gelu" or act == "relu" or act == "linear"):
                # GELU / ReLU / linear branch  : 4 d² params
                h  = 4 * d
                self.fc1 = LinearDebiased(d, h, bias=False, init_g=G_W_INIT)
                self.fc2 = LinearDebiased(h, d, bias=True,  init_g=G_W_INIT)
                #self.fc2.bias.register_hook(lambda g: g.mul_(G_B_INIT))
                self.fc2.bias.register_hook(lambda grad: grad * G_B_INIT)

                self.act_fn = {
                    "gelu":  F.gelu,
                    "relu":  F.relu,
                    "linear": lambda x: x,
                }[act]
            return
        elif not debias:
            if act == "swiglu":
                # match 8 d² params ⇒ h ≈ (8/3)d
                h = int(round(8 * d / 3))
                self.fc1 = nn.Linear(d, 2*h, bias=False)
                self.fc2 = nn.Linear(h, d, bias=True)
                # now give act_fn a proper function
                self.act_fn = lambda u: swiglu(*u.chunk(2, dim=-1))
            elif (act == "gelu" or act == "relu" or act == "linear"):
                h = 4 * d
                self.fc1 = nn.Linear(d, h, bias=False)
                self.fc2 = nn.Linear(h, d, bias=True)
                self.act_fn = {
                    "gelu":  F.gelu,
                    "relu":  F.relu,
                    "linear": lambda x: x,
                }[act]
        else:
            raise ValueError(f"Unsupported activation: {act}")

    def forward(self, x):
        z = self.ln(x)
        if self.debias:
            z_corr = debias_act(z, self.g_a1)      # activation correction
            u      = self.fc1(z_corr)
            h      = self.act_fn(u)
            h_corr = debias_act(h, self.g_a2)      # second GEMM’s activation
            return x + self.fc2(h_corr)
        else:
            u = self.fc1(z)
            h = self.act_fn(u)
            return x + self.fc2(h)

class ResidualMLP(nn.Module):
    def __init__(self, d, L, ln=True, act="gelu", debias=False):
        super().__init__()
        self.layers = nn.ModuleList([ResidualBlock(d, ln, act, debias) for _ in range(L)])
    def forward(self, x):
        for blk in self.layers: x = blk(x)
        return x

# ----------------- teacher factory ---------------------------------
class SwigLU(nn.Module):
    """Standalone SiLU‑GLU a la PaLM."""
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.silu(b)
    
def make_teacher(input_dim: int,
                 teacher_width: int,
                 depth: int,
                 act: str,
                 device: str | torch.device = "cpu") -> nn.Module:
    """
    Build a deep MLP that can mirror or differ from the student,
    with matched parameter counts even when act="swiglu". We note
    that unlike the student, the teacher does not contain a layernorm.
    """
    # base mapping for non-SwiGLU
    act_map = {
        "linear": nn.Identity(),
        "gelu":   nn.GELU(),
        "relu":   nn.ReLU(),
        "tanh":   nn.Tanh(),
    }

    act = act.lower()
    if act not in {*act_map, "swiglu"}:
        raise ValueError(f"unknown teacher activation '{act}'")

    if act == "swiglu":
        h = int(round(2 * teacher_width / 3))
        hidden_in, hidden_out = 2 * h, h
        act_mod = SwigLU()
    else:
        h = teacher_width
        hidden_in = hidden_out = h
        act_mod = act_map[act]

    def one_block():
        return [
            nn.Linear(input_dim,  hidden_in,  bias=False),
            act_mod,
            nn.Linear(hidden_out, input_dim,  bias=False),
        ]

    layers = []
    for _ in range(depth):
        layers.extend(one_block())

    teacher = nn.Sequential(*layers).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher

def build_param_slices(model, keep_ln):
    """Helper function to build slices for all parameters in the model."""
    slices, idx = [], 0
    for b, blk in enumerate(model.layers):
        for tag, m in [("ln", blk.ln), ("fc1", blk.fc1), ("fc2", blk.fc2)]:
            if tag=="ln" and not keep_ln: continue
            size = sum(p.numel() for p in m.parameters())
            slices.append((f"blk{b}_{tag}", idx, idx+size))
            idx += size
    return slices

# ===================  OPTIMIZER  =================================
def make_optimizer(model, args):
    if args.sgd:
        opt = torch.optim.SGD(model.parameters(),
                              lr=args.lr_max, momentum=args.momentum,)
    else:
        opt = torch.optim.Adam(model.parameters(),
                               lr=args.lr_max, betas=(0.9,0.999))
    sched = None
    if args.lr_schedule:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=args.steps_total, eta_min=args.lr_min)
    return opt, sched

# ===================  MX kernel flip  ============================
from mx import finalize_mx_specs, mx_mapping
def make_mx_specs(args):
    
    mx_dict = {
        "w_elem_format":      args.w_elem_format,
        "a_elem_format":      args.a_elem_format,
        "w_elem_format_bp":   args.w_elem_format_bp_w,
        "a_elem_format_bp_ex":args.a_elem_format_bp_ex,
        "a_elem_format_bp_os":args.a_elem_format_bp_os,
        "block_size": args.block_size,
        "scale_bits": args.scale_bits,
        "custom_cuda": args.use_custom_cuda, # IF YOU ARE RUNNING AN INTERVENTION EXPERIMENT OR APPLYING ANY MODIFICATIONS, LIKE BUMPED UP EXPONENT OR NON QUANTIZED LN, YOU SHOULD NOT SET THIS TO TRUE
        "quantize_backprop": True if not args.dont_quantize_backprop else False,
        "bfloat": 16,
        "bump_up_overflow_exponent": args.bump_up_overflow_exponent,
        "dont_quantize_layernorm": args.dont_quantize_layernorm,
        "dont_quantize_gelu": args.dont_quantize_gelu,
    }

    print(f"MX specs: {mx_dict}")
    
    return finalize_mx_specs(mx_dict)

# ---------- run‑folder helper ------------------------------------
def make_save_path(arg_obj, save_root):
    """
    Return   <out_root>/<descriptive_folder_name>

    * arg_obj can be either a Namespace or the dict from vars(args)
    * only whitelists the hyper‑params that define a run
    * keys are sorted → deterministic path
    """
    cfg = vars(arg_obj) if not isinstance(arg_obj, dict) else arg_obj
    out_root = save_root #pathlib.Path(cfg["out"]).expanduser().resolve()

    # whitelist (edit to taste) - what will be used to name the run folder
    keep_keys = {
        "depth"        : ("L",  "{}"),
        "width"        : ("D",  "{}"),
        "batch"        : ("bs", "{}"),
        "lr_max"       : ("lr", "{:.0e}"),
        "w_elem_format"  : ("fmt","{}"),
        "scale_bits"   : ("sb", "{}"),
        "sgd"          : ("sgd",""),
        "lr_schedule"  : ("cos",""),
        "no_ln"        : ("noln",""),
        "no_act"       : ("lin",""),
        "act"          : ("act","{}"),
        "seed"         : ("s",  "{}"),
        "w_elem_format_bp_w": ("bp","{}"),
        "a_elem_format": ("a_fmt","{}"),
        "a_elem_format_bp_ex": ("a_bp_ex","{}"),
        "a_elem_format_bp_os": ("a_bp_os","{}"),
    }

    parts = []
    for k in sorted(keep_keys):
        if k not in cfg:
            continue
        prefix, fmt = keep_keys[k]
        val = cfg[k]
        if isinstance(val, bool):
            if val:
                parts.append(prefix)
        else:
            parts.append(f"{prefix}{fmt.format(val)}")
    
    # generate random number between 1 and 10000
    # todo - change this to uuid
    random_number = random.randint(1, 1000000)

    folder_name = "_".join(parts) + "_" + str(random_number)
    return out_root / folder_name

_KEEP_KEYS = {
    "depth"       : ("L",  "{}"),
    "width"       : ("D",  "{}"),
    "batch"       : ("bs", "{}"),
    "lr_max"      : ("lr", "{:.0e}"),
    "w_elem_format" : ("fmt","{}"),
    "sgd"         : ("sgd",""),
    "no_ln"       : ("noln",""),
    "no_act"      : ("lin",""),
    "act"         : ("act","{}"),
    "dont_quantize_backprop": ("dqbp",""),
    "seed"        : ("s",  "{}"),
    "w_elem_format_bp_w": ("bp","{}"),
    "a_elem_format": ("a_fmt","{}"),
    "a_elem_format_bp_ex": ("a_bp_ex","{}"),
    "a_elem_format_bp_os": ("a_bp_os","{}"),

}

def custom_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=0.5)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_tags(cfg: dict, *, random_tag: bool = True, save_root: str = None):
    """Return (tag_string, folder_path)"""
    parts = []
    for k in sorted(_KEEP_KEYS):
        if k not in cfg:
            continue
        prefix, fmt = _KEEP_KEYS[k]
        val = cfg[k]
        if isinstance(val, bool):
            if val:
                parts.append(prefix)
        else:
            parts.append(f"{prefix}{fmt.format(val)}")

    tag = "_".join(parts)
    if random_tag:
        #tag_rand = f"{random.randint(0,9999):04d}"
        import uuid
        tag_rand = uuid.uuid4().hex[:4]
        folder = save_root / f"{tag}_{tag_rand}"
    else:
        folder = save_root / tag
    return tag, folder

def mx_safe_norm(t: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(t**2))

def save_instability_tensors(tensor, export_dir, global_step, blk_idx, name, step_threshold_min, step_threshold_max):
    """
    A useful helper function to call around when you have an instability to save stuff.
    """
    if global_step >= step_threshold_min and global_step <= step_threshold_max:
        torch.save(tensor.cpu(), export_dir/f"step{global_step:05d}_blk{blk_idx:02d}_{name}.pt" )
    else:
        pass



# =================== TRAINING LOOPS FOR FP32 AND MX ================================
@dataclass
class LoopData:
    args: argparse.Namespace
    global_step: int = 0                     # shared WandB/x-axis counter

    # ─── caches produced during FP32 phase ────────────────────────────────
    g_fp32_mean: list = field(default_factory=list)
    g_fp32_norm: list = field(default_factory=list)
    group_mean_fp32: dict = field(default_factory=lambda: defaultdict(dict))
    g32_store: dict = field(default_factory=dict)

    # ─── records common to both phases ───────────────────────────────────
    loss_rec: dict = field(default_factory=lambda: {"fp32": {}, "mx": {}})
    eps_rec:  dict = field(default_factory=dict)

    # ─── run dir ────────────────────────────────────────────────
    run_dir: pathlib.Path = field(default_factory=pathlib.Path)

    # convenience wrapper so we never forget the step kwarg
    def log(self, metrics: dict):
        wandb.log(metrics, step=self.global_step)

def save_checkpoint(step,
                    model,
                    opt,
                    sched,
                    args,
                    folder: pathlib.Path,
                    mode="mx",
                    ld: "LoopData | None" = None):
    """
    Save everything required to restart training **and** to compute
    epsilon metrics later on.

    If `ld` is provided we also store the FP-32 caches that live in it.
    """
    folder = pathlib.Path(folder); folder.mkdir(parents=True, exist_ok=True)

    payload = {
        "global_step": step,
        "model_state": model.state_dict(),
        "opt_state":   opt.state_dict()   if opt   is not None else None,
        "sched_state": sched.state_dict() if sched is not None else None,
        "rng_state":   torch.get_rng_state(),
        "cuda_rng":    torch.cuda.get_rng_state()
                         if torch.cuda.is_available() else None,
        "args":        vars(args),
        "mode":        mode,
        "original_run_dir": str(ld.run_dir) if ld is not None else None,
        "steps_total": args.steps_total,
    }

    if ld is not None:               # ← NEW
        payload.update({
            "g_fp32_mean":     ld.g_fp32_mean,
            "g_fp32_norm":     ld.g_fp32_norm,
            "group_mean_fp32": dict(ld.group_mean_fp32),   # defaultdict→dict
            "g32_store":       {k: v.cpu()
                                for k, v in ld.g32_store.items()},
        })

    torch.save(payload, folder / f"ckpt_{mode}_step{step:04d}.pt")

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # don't actually think below is needed anymore, but keep for safety
    rng_state = ckpt.get("rng_state", None)
    if rng_state is not None and not isinstance(rng_state, torch.ByteTensor):
        rng_state = torch.ByteTensor(rng_state.cpu() if hasattr(rng_state, 'cpu') else rng_state)
    torch.set_rng_state(rng_state)
    
    cuda_rng = ckpt.get("cuda_rng", None)
    if torch.cuda.is_available() and cuda_rng is not None:
        if isinstance(cuda_rng, list):
            for dev, state in enumerate(cuda_rng):
                if dev < torch.cuda.device_count():
                    if not isinstance(state, torch.ByteTensor):
                        state = torch.ByteTensor(state.cpu() if hasattr(state, 'cpu') else state)
                    with torch.cuda.device(dev):
                        torch.cuda.set_rng_state(state)
        else:
            if not isinstance(cuda_rng, torch.ByteTensor):
                cuda_rng = torch.ByteTensor(cuda_rng.cpu() if hasattr(cuda_rng, 'cpu') else cuda_rng)
            torch.cuda.set_rng_state(cuda_rng)
    
    return ckpt


# ───────────────────────── FP32 TRAINING PHASE ─────────────────────────────
def fp32_loop(ld: LoopData,
              model: nn.Module,
              teacher_fp32: nn.Module,
              batches: list[torch.Tensor],
              val_batches: list[torch.Tensor],
              param_slices):
    """
    Runs the full-precision warm-up phase.
    All metrics are written into `ld` and WandB.
    """
    args = ld.args
    opt32, sched32 = make_optimizer(model, args)
    if args.save_checkpoints:
        checkpoint_dir = pathlib.Path(ld.run_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step, x in enumerate(batches):
        x = x.to(args.device, non_blocking=True)
        y = teacher_fp32(x) + args.noise_std * torch.randn_like(x, dtype=x.dtype)
        opt32.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()

        if args.log_weight_clipping:
            clipped_total_h_fp32 = elems_total_h_fp32 = 0
            clipped_total_out_fp32 = elems_total_out_fp32 = 0

            with torch.no_grad():
                # Record overflow values for the weights and activations - for fp32 these
                # should basically be 0 but just sanity check.
                z = x
                post_ln_acts = []
                for blk_idx, blk in enumerate(model.layers):
                    x_in = z
                    z_norm = blk.ln(x_in) if not args.no_ln else x_in
                    post_ln_acts.append(z_norm)
                    a1 = blk.fc1(z_norm)
                    h  = blk.act_fn(a1)
                    c_h, t_h = count_clipped_values_fp32(h)
                    ld.log({f"ClipAct/fp32_blk{blk_idx}_hidden_clipped_frac": c_h / t_h})
                    clipped_total_h_fp32 += c_h;  elems_total_h_fp32 += t_h
                    ld.log({
                        f"act/mean_hid{blk_idx}_fp32": h.mean().item(),
                        f"act/var_hid{blk_idx}_fp32":  h.var(unbiased=False).item(),
                    })
                    # alignment metric - how well the hidden layer aligns with the weights
                    W2    = blk.fc2.weight
                    proj  = torch.matmul(W2, h.T).abs()
                    numer = proj.mean(dim=1)
                    w_norm = (W2**2).sum(dim=1, keepdim=True).sqrt()
                    h_norm = (h**2).sum(dim=1, keepdim=True).sqrt().T
                    denom  = (w_norm * h_norm).mean(dim=1) + 1e-12
                    align  = (numer / denom.squeeze()).mean()
                    ld.log({f"align/fp32_blk{blk_idx}": align.item()})
                    z     = x_in + blk.fc2(h)
                    c_o, t_o = count_clipped_values_fp32(z)
                    ld.log({f"ClipAct/fp32_blk{blk_idx}_output_clipped_frac": c_o / t_o})
                    clipped_total_out_fp32 += c_o;  elems_total_out_fp32 += t_o
                    # LN gamma + beta stats
                    gamma = blk.ln.weight.data.detach().cpu()
                    beta  = blk.ln.bias.data.detach().cpu()
                    ld.log({
                        f"act/fp32_ln{blk_idx}_gamma_mean": gamma.mean().item(),
                        f"act/fp32_ln{blk_idx}_gamma_var":  gamma.var().item(),
                        f"act/fp32_ln{blk_idx}_beta_mean":  beta.mean().item(),
                        f"act/fp32_ln{blk_idx}_beta_var":   beta.var().item(),
                    })
                    c_g, t_g = count_clipped_values_fp32(gamma)
                    c_b, t_b = count_clipped_values_fp32(beta)
                    ld.log({
                        f"clipW/fp32_ln{blk_idx}_gamma_clipped_frac": c_g / t_g,
                        f"clipW/fp32_ln{blk_idx}_beta_clipped_frac":  c_b / t_b,
                    })
                if elems_total_h_fp32:
                    ld.log({"ClipAct/fp32_total_frac_hidden_all_layers":
                            clipped_total_h_fp32 / elems_total_h_fp32})
                if elems_total_out_fp32:
                    ld.log({"ClipAct/fp32_total_frac_output_all_layers":
                            clipped_total_out_fp32 / elems_total_out_fp32})
                for k, a in enumerate(post_ln_acts):
                    ld.log({
                        f"act/mean_layer{k}_input_after_ln_fp32": a.mean().item(),
                        f"act/var_layer{k}_input_after_ln_fp32":  a.var(unbiased=False).item(),
                    })

        # gradient bookkeeping
        g32 = grad_vec(model, drop_scalars=True).cpu()
        if args.store_full_gradients and step % STORE_GRADS_EVERY == 0:
            ld.g32_store[step] = g32.clone().float()
        ld.g_fp32_mean.append(g32.mean().item())
        ld.g_fp32_norm.append(torch.norm(g32).item())
        for name, lo, hi in param_slices:
            ld.group_mean_fp32[step][name] = g32[lo:hi].mean().item()
        del g32

        # optimiser step & logs
        opt32.step();  sched32.step() if sched32 else None
        ld.log({"train/loss_fp32": loss.item(),
                "lr_fp32": sched32.get_last_lr()[0] if sched32 else args.lr_max})
        ld.loss_rec["fp32"][step] = loss.item()

        if step % args.val_every == 0:
            model.eval()
            v = val_loss(model, teacher_fp32, F.mse_loss, val_batches)
            ld.loss_rec["fp32"][f"val_{step}"] = v
            print(f"val_loss_fp32: {v:.3e}")
            ld.log({"val/loss_fp32": v})
            model.train()

        if (args.save_checkpoints and args.checkpoint_window_center is not None \
            and abs(ld.global_step - args.checkpoint_window_center) <= args.checkpoint_window_size) \
            and (ld.global_step % args.checkpoint_every == 0 or ld.global_step == args.steps_total - 1):
            save_checkpoint(ld.global_step, model, opt32, sched32, args, pathlib.Path(checkpoint_dir), mode="fp32", ld=ld)

        ld.global_step += 1


# ───────────────────────── MX TRAINING PHASE ───────────────────────────────
def mx_loop(ld: LoopData,
            model_init_state: dict,
            teacher_fp32: nn.Module,
            batches: list[torch.Tensor],
            val_batches: list[torch.Tensor],
            param_slices,
            export_dir: pathlib.Path,
            model = None,
            opt = None,
            sched = None,
            targets = None,
            val_targets = None,
            custom_cuda_flag: bool = True,
            inject_mx_ops: bool = True):
    """
    Runs the low-precision phase **with identical logging**.
    Reads FP32 reference stats from `ld`, writes MX stats back into `ld`.
    The last 3 args are only used for the intervention experiment when a checkpointed
    model and optimizer and lr sched state is passed in.
    """
    args = ld.args
    device  = torch.device(args.device)
    act_name = "linear" if args.no_act else args.act
    # set_seed(ld.args.seed)

    if args.save_checkpoints:
        checkpoint_dir = pathlib.Path(ld.run_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # rebuild the MX student & optimiser
    if model is None and opt is None and sched is None:
        if args.debias_MX_gradient:
            model = ResidualMLP(args.width, args.depth,
                                not args.no_ln, act_name, debias=True).to(device)
            model.load_state_dict(model_init_state, strict=False)
        else:
            model = ResidualMLP(args.width, args.depth,
                                not args.no_ln, act_name).to(device)
            model.load_state_dict(model_init_state)

        opt, sched = make_optimizer(model, args)

    if inject_mx_ops:
        mx_mapping.inject_pyt_ops(make_mx_specs(args))

    fp32_steps = ld.args.steps_total
    print(f"MX phase: {fp32_steps} FP32 steps, {len(batches)} batches, {len(val_batches)} val batches")
    base_mx_step = ld.global_step - fp32_steps
    print(f"MX phase: base step {base_mx_step} (global step {ld.global_step})")

    for step, x in enumerate(batches):
        x = x.to(device)
        rel_step = base_mx_step + step
        # print(f"MX step {rel_step} (global step {ld.global_step})")

        if targets is not None:
            y = targets[step]
            y = y.to(device)
        else:
            y = teacher_fp32(x) + args.noise_std * torch.randn_like(x, dtype=x.dtype)
        opt.zero_grad(set_to_none=True)
        loss_mx = F.mse_loss(model(x), y); loss_mx.backward()

        if args.log_weight_clipping:
            clipW_tot = elemsW_tot = 0
            for n, p in model.named_parameters():
                if ".weight" in n:
                    c, t = count_clipped_values(p.data, 32, args.w_elem_format) # assume w_format for counting clipped
                    ld.log({f"clipW/{n}": c / t});  clipW_tot += c;  elemsW_tot += t
            if elemsW_tot:
                ld.log({"clipW/total_frac": clipW_tot / elemsW_tot})

        # gradient vec & eps metrics
        g_mx = grad_vec(model, drop_scalars=True).cpu()

        if args.debias_MX_gradient:
            for n, p in model.named_parameters():
                if n.endswith(".g"):            ld.log({f"g/{n}": p.item()})
                if n.endswith(("g_a1", "g_a2")): ld.log({f"g/act_{n}": p.item()})
        
        # log the gradient vector, epsilon statistics defined in paper
        mx_mean = g_mx.mean().item()
        mx_norm = ((g_mx**2).sum().sqrt()).item()
        eps_mean = mx_mean - ld.g_fp32_mean[rel_step]
        eps_norm_upper_bound = mx_norm + ld.g_fp32_norm[rel_step]

        group_eps_mean = {}
        for name, lo, hi in param_slices:
            mx_grp_mean  = g_mx[lo:hi].mean().item()
            fp32_grp_mean = ld.group_mean_fp32[rel_step][name]
            group_eps_mean[name] = mx_grp_mean - fp32_grp_mean
            ld.log({f"eps/mean_{name}": group_eps_mean[name]})

        # ε-vector diagnostics (only if we kept g32_store)
        if args.store_full_gradients and rel_step in ld.g32_store:
            eps_vec   = g_mx - ld.g32_store[rel_step].cpu()
            eps_norm2 = (eps_vec**2).sum().item()
            g_norm2   = (ld.g32_store[rel_step]**2).sum().item()
            cos_theta = torch.dot(g_mx, ld.g32_store[rel_step].cpu()) / (mx_norm * ld.g_fp32_norm[rel_step] + 1e-12)
            sigma2    = eps_norm2 / (g_norm2 + 1e-12)
            ld.log({
                "stab/cos": cos_theta,
                "stab/rhs_2_over_sigma2": 2.0 / (sigma2 + 1e-12),
                "stab/zeta_norm": sigma2,
            })
            # clean up to save GPU RAM
            del ld.g32_store[rel_step]

        # the big block below is just doing a couple of things:
        # 1. logging the overflow statistics
        # 2. logging some activation and LN statistics
        # 3. saving tensors at avrious points in the model for understanding where instability happens
        # you should comment out `save_instability_tensors` calls if you don't need to save the tensors
        if args.log_weight_clipping:
            clipped_total_h, elems_total_h = 0, 0
            clipped_total_layer, elems_total_layer = 0, 0
            with torch.no_grad():
                z = x
                input_after_layernorm_if_ln = []
                hs = []
                for blk_idx, blk in enumerate(model.layers):
                    x_in = z

                    save_instability_tensors(x_in, export_dir, ld.global_step, blk_idx, "input", 12650, 12800)

                    z_norm = blk.ln(x_in) if not args.no_ln else x_in

                    save_instability_tensors(z_norm, export_dir, ld.global_step, blk_idx, "postLN", 12650, 12800)

                    input_after_layernorm_if_ln.append(z_norm)

                    a1 = blk.fc1(z_norm)
                    h  = blk.act_fn(a1)

                    save_instability_tensors(h, export_dir, ld.global_step, blk_idx, "hidden", 12650, 12800)


                    hs.append(h)

                    W2 = blk.fc2.weight
                    proj = torch.matmul(W2, h.T).abs()
                    numer = proj.mean(dim=1)
                    w_norm = (W2 ** 2).sum(dim=1, keepdim=True).sqrt()
                    h_norm = (h  ** 2).sum(dim=1, keepdim=True).sqrt().T
                    denom  = (w_norm * h_norm).mean(dim=1) + 1e-12        
                    align  = (numer / denom.squeeze()).mean()
                    wandb.log({f"align/blk{blk_idx}": align.item()}, step=ld.global_step)

                    # hidden state:
                    c_h, t_h = count_clipped_values(h, 32, args.a_elem_format)
                    wandb.log({f"ClipAct/blk{blk_idx}_hidden_clipped_frac": c_h / t_h}, step=ld.global_step)
                    clipped_total_h += c_h;  elems_total_h += t_h

                    wandb.log({f"act/mean_hid{blk_idx}": h.mean().item(),
                            f"act/var_hid{blk_idx}":  h.var(unbiased=False).item()}, step=ld.global_step)

                    # layer output after adding to residual stream
                    z_blk = blk.fc2(h) + x_in
                    c_z, t_z = count_clipped_values(z_blk, 32, args.w_elem_format)
                    wandb.log({f"ClipAct/blk{blk_idx}_output_clipped_frac": c_z / t_z}, step=ld.global_step)
                    clipped_total_layer += c_z;  elems_total_layer += t_z

                    z = z_blk

                    # log layernorm data
                    gamma = blk.ln.weight.data.detach().cpu()
                    beta  = blk.ln.bias.data.detach().cpu()

                    wandb.log({
                        f"act/ln{blk_idx}_gamma_mean": gamma.mean().item(),
                        f"act/ln{blk_idx}_gamma_var":  gamma.var().item(),
                        f"act/ln{blk_idx}_beta_mean":  beta.mean().item(),
                        f"act/ln{blk_idx}_beta_var":   beta.var().item(),
                    }, step=ld.global_step)

                    # if global set if > 12650 and < 12800, save activaitons and gamma and beta
                    # these steps are hard-coded in for now... [TODO: make them args]
                    save_instability_tensors(gamma, export_dir, ld.global_step, blk_idx, "gamma", 12650, 12800)
                    save_instability_tensors(beta, export_dir, ld.global_step, blk_idx, "beta", 12650, 12800)


                    # log how much of gamma and beta are being clipped
                    c_gamma, t_gamma = count_clipped_values(gamma, 32, args.w_elem_format)
                    wandb.log({f"clipW/ln{blk_idx}_gamma_clipped_frac": c_gamma / t_gamma}, step=ld.global_step)
                    c_beta, t_beta = count_clipped_values(beta, 32, args.w_elem_format)
                    wandb.log({f"clipW/ln{blk_idx}_beta_clipped_frac": c_beta / t_beta}, step=ld.global_step)

                if elems_total_h:
                    wandb.log({"ClipAct/total_frac_hidden_all_layers": clipped_total_h / elems_total_h}, step=ld.global_step)
                if elems_total_layer:
                    wandb.log({"ClipAct/total_frac_layer_all_layers": clipped_total_layer / elems_total_layer}, step=ld.global_step)
                
                for k,a in enumerate(input_after_layernorm_if_ln):
                    wandb.log({f"act/mean_layer{k}_input_after_ln": a.mean().item(),
                            f"act/var_layer{k}_input_after_ln":  a.var(unbiased=False).item()}, step=ld.global_step)
                    

            model.train()

        # bookkeeping
        ld.loss_rec["mx"][step] = loss_mx.item()
        ld.eps_rec[step] = {
            "mean": eps_mean,
            "norm_upper": eps_norm_upper_bound,
            "mx_norm": mx_norm,
            "fp32_norm": ld.g_fp32_norm[rel_step],
            "group_mean_fp32": group_eps_mean.copy(),
        }

        ld.log({
            "eps_step": step,
            "eps/mean": eps_mean,
            "eps/norm_upper_bound": eps_norm_upper_bound,
            "train/loss_mx": loss_mx.item(),
            "lr": sched.get_last_lr()[0] if sched else args.lr_max,
        })

        opt.step();  sched.step() if sched else None

        if step % args.val_every == 0:
            model.eval()
            #v = val_loss(model, teacher_fp32, F.mse_loss, val_batches)
            if val_targets is not None:
                losses = [F.mse_loss(model(xv.to(device)), yv.to(device)) for xv, yv in zip(val_batches, val_targets)]
                v = torch.stack(losses).mean().item()
            else:
                v = val_loss(model, teacher_fp32, F.mse_loss, val_batches)

            ld.loss_rec["mx"][f"val_{step}"] = v
            print(f"val_loss_mx: {v:.3e}")
            ld.log({"val/loss_mx": v})
            model.train()

        ld.global_step += 1
        if (args.save_checkpoints and args.checkpoint_window_center is not None \
            and abs(ld.global_step - args.checkpoint_window_center) <= args.checkpoint_window_size) \
            and (ld.global_step % args.checkpoint_every == 0 or ld.global_step == args.steps_total - 1):
            
            save_checkpoint(ld.global_step, model, opt, sched, args, pathlib.Path(checkpoint_dir), mode="mx", ld = ld)

        
def run_dual(args):
    print("Setting seed")
    set_seed(args.seed)
    device = torch.device(args.device)
    e_max, v_max = fmt_meta(args.w_elem_format) # use weights elem format to get this info

    args.save_root.mkdir(parents=True, exist_ok=True)

    # build run directory & WandB tag
    cfg = vars(args)
    tag, run_dir = build_tags(cfg, random_tag=True, save_root=args.save_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    export_dir = run_dir / "instability_dumps"
    export_dir.mkdir(exist_ok=True)

    import uuid
    wandb_tag = f"{tag}_{uuid.uuid4().hex[:4]}"
    wandb.init(
        project=args.wandb_project,
        name=wandb_tag,
        id=wandb_tag,
        config=cfg,
        resume="allow",
        settings=wandb.Settings(init_timeout=120),
    )

    # Model & Teacher
    print("Getting model and teacher")
    act_name = "linear" if args.no_act else args.act
    model = ResidualMLP(args.width, args.depth, not args.no_ln, act_name).to(device)
    if args.use_custom_init:
        model.apply(custom_init)
    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    teacher = make_teacher(
        input_dim=args.width,
        teacher_width=args.teacher_width,
        depth=args.teacher_depth,
        act=args.teacher_act,
        device="cpu",
    )
    teacher_fp32 = copy.deepcopy(teacher).float().to(device)

    # Data & Slices
    # Do it on CPU first, then move to GPU
    # We save the dataset to disk so that we can re-use it later in case we do an intervention
    # experiment.  Previously was doing this on-the-fly with fixed seeds, but this seems more
    # safe and reproducible.

    print("Generating synthetic dataset")
    # batches = [gaussian(args.batch, args.width, device) for _ in range(args.steps_total)]
    # val_batches = [gaussian(args.val_batch, args.width, device) for _ in range(args.val_steps)]

    device_cpu   = torch.device("cpu")
    device_model = torch.device(args.device)

    train_x = [gaussian(args.batch,  args.width, device_cpu) for _ in range(args.steps_total)]
    val_x   = [gaussian(args.val_batch, args.width, device_cpu) for _ in range(args.val_steps)]

    with torch.no_grad():
        train_y = [teacher_fp32(x.to(device_model)).cpu() + args.noise_std * torch.randn_like(x) for x in train_x]
        val_y   = [teacher_fp32(x.to(device_model)).cpu() + args.noise_std * torch.randn_like(x) for x in val_x]

    X = torch.stack(train_x, dim=0)   # shape (steps_total, batch, width)
    Y = torch.stack(train_y, dim=0)
    Vx = torch.stack(val_x,   dim=0)
    Vy = torch.stack(val_y,   dim=0)
    torch.save({"train_x": X, "train_y": Y, "val_x": Vx, "val_y": Vy}, run_dir/"dataset.pt")


    batches     = [X[i]  for i in range(X.shape[0])]
    val_batches = [Vx[i] for i in range(Vx.shape[0])]

    param_slices = build_param_slices(model, not args.no_ln)

    # ─── Run FP32 Warm-up ─────────────────────────────────────────────
    ld = LoopData(args=args, run_dir=run_dir)
    fp32_loop(ld, model, teacher_fp32, batches, val_batches, param_slices)

    # ─── Run MX Phase ────────────────────────────────────────────────
    print("MX phase")
    mx_loop(ld, init_state, teacher_fp32, batches, val_batches, param_slices, export_dir, inject_mx_ops=(not args.dont_inject_mx_ops))

    # ─── Clean up ───────────────────────────────────────────────────
    wandb.finish()

def merge_args(base: dict, override: argparse.Namespace) -> argparse.Namespace:
    """Return a Namespace equal to `base` but with any non-default
    values from `override` patched in.  We ignore meta flags such as
    run_intervention / intervention_checkpoint."""
    merged = argparse.Namespace(**base)
    skip = {"run_intervention", "intervention_checkpoint", "steps_total"}
    for k, v in vars(override).items():
        if k in skip:                   # never propagate
            continue
        if v != getattr(merged, k, None):
            setattr(merged, k, v)
    return merged


def run_intervention(args, custom_cuda_flag=False):
    """
    Resume from a checkpoint and continue *only* the MX phase, possibly
    with a modified MX spec (exponent bump, LayerNorm in FP32, …).
    """
    #set_seed(args.seed)
    device = torch.device(args.device)
    ckpt   = load_checkpoint(args.intervention_checkpoint, device=device)
    # state_cpu  = torch.get_rng_state().clone()
    # state_cuda = (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


    # we want to ensure that we are picking up the training in EXACTLY the same way as before,
    # modulo the intervention changes which will really only change the `mx_specs` object
    # we also need to make sure to set the random seed and that we pick up at the batch we left off at,
    # so that there is parity.

    # ───────────────────────────────────────────────────────── base config
    base_args = merge_args(ckpt["args"], args)     # apply overrides
    base_args.device = args.device                 # honour current CLI
    # set_seed(base_args.seed)

    # ─────────────────────────────────────────────────────────  W&B set-up
    tag_suffix = []
    if base_args.bump_up_overflow_exponent: tag_suffix.append("bumpExp")
    if base_args.dont_quantize_layernorm:   tag_suffix.append("noLNq")
    if base_args.dont_quantize_gelu:        tag_suffix.append("noGELUq")
    if base_args.a_elem_format == "bfloat16":tag_suffix.append("bf16Act")

    new_tag = "_".join(tag_suffix) or "control"

    wandb.init(
        project   = base_args.wandb_project,
        name      = f"intervention_{new_tag}_{int(time.time())}",
        config    = vars(base_args),
        resume    = "allow",
        settings  = wandb.Settings(init_timeout=120),
    )

    
    
    orig_dir = pathlib.Path(ckpt["original_run_dir"])
    data = torch.load(orig_dir / "dataset.pt")

    X, Y, Vx, Vy = data["train_x"], data["train_y"], data["val_x"], data["val_y"]



    fp32_steps   = base_args.steps_total
    rel_step     = ckpt["global_step"] - fp32_steps   # MX steps already done
    print(f"FP32 steps: {fp32_steps}, MX steps already done: {rel_step} (global step {ckpt['global_step']})")

    train_tail_x = X[rel_step:]
    train_tail_y = Y[rel_step:]
    val_batches  = [Vx[i] for i in range(Vx.shape[0])]   # or keep as a tensor
    val_targets  = [Vy[i] for i in range(Vy.shape[0])]

    act_name = "linear" if base_args.no_act else base_args.act
    print(f"Loading model from {args.intervention_checkpoint}")
    if base_args.debias_MX_gradient:
        model = ResidualMLP(base_args.width, base_args.depth,
                            not base_args.no_ln, act_name,
                            debias=True).to(device)
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model = ResidualMLP(base_args.width, base_args.depth,
                            not base_args.no_ln, act_name).to(device)
        model.load_state_dict(ckpt["model_state"])

    opt, sched = make_optimizer(model, base_args)
    if ckpt["opt_state"] is not None:
        print(f"Loading optimizer state from {args.intervention_checkpoint}")
        opt.load_state_dict(ckpt["opt_state"])
    if ckpt["sched_state"] is not None and sched is not None:
        print(f"Loading scheduler state from {args.intervention_checkpoint}")
        sched.load_state_dict(ckpt["sched_state"])

    print("Checkpoint global step:", ckpt["global_step"])
    ld = LoopData(args=base_args,
                  global_step=ckpt["global_step"],
                  g_fp32_mean = ckpt["g_fp32_mean"],
                  g_fp32_norm = ckpt["g_fp32_norm"],
                  group_mean_fp32 = defaultdict(dict, ckpt["group_mean_fp32"]),
                  g32_store = {int(k): v.to(device)
                               for k, v in ckpt.get("g32_store", {}).items()}
                 )

    # IMPORTANT:  we do **not** have the FP32 caches anymore, so the
    # ε-metrics cannot be computed.  We simply turn them off by pretending
    # `store_full_gradients=False`.
    base_args.store_full_gradients = bool(ld.g32_store)

    param_slices = build_param_slices(model, not base_args.no_ln)

    export_dir = ld.run_dir / "intervention_dumps"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting MX loop at intervened step.  Number of MX batches to process: {len(train_tail_x)} with {len(train_tail_y)} targets.")
    mx_loop(ld,
            ckpt["model_state"],
            None,
            train_tail_x,
            val_batches,
            param_slices,
            export_dir,
            model=model,
            opt=opt,
            sched=sched,
            targets=train_tail_y, val_targets=val_targets,
            custom_cuda_flag=custom_cuda_flag,
            inject_mx_ops=(not base_args.dont_inject_mx_ops),)

    wandb.finish()

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--teacher_width", type=int, default=None)
    p.add_argument("--teacher_depth", type=int, default=None)
    p.add_argument("--teacher_act",
                   choices=["gelu", "relu", "swiglu", "tanh"],
                   default=None,help="Activation used inside the teacher; defaults to the student’s --act")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--steps_total", type=int, default=7000)
    p.add_argument("--lr_max", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--lr_schedule", action="store_true")
    p.add_argument("--sgd", action="store_true",
               help="use plain SGD (no momentum) instead of Adam")
    p.add_argument("--momentum", type=float, default=0.0, help="SGD momentum")

    p.add_argument("--w_elem_format", default="fp8_e4m3")
    p.add_argument("--a_elem_format", default="fp8_e4m3")
    p.add_argument("--w_elem_format_bp_w", default="fp8_e4m3")
    p.add_argument("--a_elem_format_bp_ex", default="fp8_e4m3")
    p.add_argument("--a_elem_format_bp_os", default="fp8_e4m3")
    p.add_argument("--block_size", type=int, default=32,)

    p.add_argument("--debias_MX_gradient", action="store_true", help="Debias MX gradients")
    p.add_argument("--log_weight_clipping", action="store_true", help="Log when weights clip in low precision")
    p.add_argument("--use_custom_init", action="store_true", help="use custom init with lower gain")
    p.add_argument("--dont_quantize_backprop", action="store_true")
    p.add_argument("--use_custom_cuda", action="store_true",
               help="Use custom CUDA kernels for MX blocks (default: use PyTorch ops)")

    # MX modification options
    p.add_argument("--bump_up_overflow_exponent", action="store_true",
               help="Apply the exponent bump trick to prevent MX blocks that flow into largest buckets.")
    p.add_argument("--dont_quantize_layernorm", action="store_true",
               help="Don't quantize the layernorm weights and biases to the same format as the activations.")
    p.add_argument("--dont_quantize_gelu", action="store_true",
               help="Don't quantize the Gelu activation.")

    p.add_argument("--scale_bits", type=int, default=8)
    p.add_argument("--act", choices=["gelu","relu","swiglu"], default="gelu")
    p.add_argument("--no_act", action="store_true")
    p.add_argument("--no_ln",  action="store_true")
    p.add_argument("--store_full_gradients", action="store_true", help="store fp32 gradients for stability analysis")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--val_batch", type=int, default=2048,
               help="validation batch size (same width)")
    p.add_argument("--val_steps", type=int, default=4,
               help="# different fixed val batches to average over")
    p.add_argument("--noise_std", type=float, default=0.005)
    p.add_argument("--out", default="eps_runs")
    p.add_argument("--wandb_project", default="mx_eps_dual_with_fp64_2")
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--save_root", type=str, default="/n/netscratch/kempner_dev/Lab/nikhilanand/eps_sweeps_depth_width")


    # in case we want to log checkpoints around the instability point
    p.add_argument("--save_checkpoints", action="store_true",
               help="Save checkpoints during the run, useful for intervention experiments.")
    p.add_argument("--checkpoint_window_center", type=int, default=6000, help="Which step is the center (we will save +/- radius with this step at the center)")
    p.add_argument("--checkpoint_window_size", type=int, default=100,
               help="Size of the window around the center checkpoint to save.")
    p.add_argument("--checkpoint_every", type=int, default=5)

    # intervention experiment - this will overwrite some of the above args if chosen
    p.add_argument("--run_intervention", action="store_true",
               help="Run the intervention experiment starting at a specified checkpoint.")
    p.add_argument("--intervention_checkpoint", type=str, default=None,
               help="Path to the checkpoint to start the intervention experiment from.")
    p.add_argument("--dont_inject_mx_ops", action="store_true",
               help="If set, do not inject MX ops into the model.")
    # -------------------------------------------------------------------------------

    args = p.parse_args()

    args.save_root = pathlib.Path(args.save_root).expanduser()


    if args.teacher_width is None:
        args.teacher_width = args.width
    
    if args.teacher_depth is None:
        args.teacher_depth = args.depth
    

    args.teacher_act = (args.teacher_act if args.teacher_act is not None
                        else ("linear" if args.no_act else args.act))

    return args

if __name__ == "__main__":
    parsed_args = parse()

    if parsed_args.run_intervention:
        custom_cuda_flag=False
        if parsed_args.intervention_checkpoint is None:
            raise ValueError("If --run_intervention is set, you must provide --intervention_checkpoint.")
        run_intervention(parsed_args, custom_cuda_flag)
    
    else:
        run_dual(parse())
