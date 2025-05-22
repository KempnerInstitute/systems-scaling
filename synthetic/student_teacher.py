import argparse, pathlib, random, csv, time, yaml, math, warnings
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import wandb
from torch.autograd import Function
from torch import Tensor

import copy
import uuid

import math
LN2        = math.log(2.0)
G_W_INIT   = LN2 ** 3
G_B_INIT   = LN2 ** 2 


# HELPER FUNCTIONS

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def gaussian(batch, dim, device="cpu"): return torch.randn(batch, dim, device=device)

def grad_vec(model, dtype=None, *, drop_scalars=False):
    """Returns a vector representation of the gradient."""
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

def val_loss(model, teacher, crit, val_batches):
    """Compute validation loss."""
    with torch.no_grad():
        losses = [crit(model(v), teacher(v)) for v in val_batches]
    return torch.stack(losses).mean().item()

def build_param_slices(model, keep_ln):
    slices, idx = [], 0
    for b, blk in enumerate(model.layers):
        for tag, m in [("ln", blk.ln), ("fc1", blk.fc1), ("fc2", blk.fc2)]:
            if tag=="ln" and not keep_ln: continue
            size = sum(p.numel() for p in m.parameters())
            slices.append((f"blk{b}_{tag}", idx, idx+size))
            idx += size
    return slices

def custom_init(m):
    """Perform custom initialisation of lower variance model parameters."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=0.5)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ===================  (Experimental) Debiasing functions  =====================================
# the below functions attempt to correct for STE bias in the backward pass
# these are still WIP so it is recommened you ignore them (they are default ignored)
# unless you set the --debias_MX_gradient flag

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
    gradient by the scalar g_a in the backward.
    """
    @staticmethod
    def forward(ctx, x: Tensor, g_a: Tensor) -> Tensor:
        ctx.save_for_backward(g_a)
        return x

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (g_a,) = ctx.saved_tensors
        return grad_out * g_a, None

def debias_act(x: Tensor, g_a: nn.Parameter) -> Tensor:
    # identical value, corrected gradient
    return ActDebiasGate.apply(x, g_a)

# ===================  MODEL  =====================================
def swiglu(x):
    a,b = x.chunk(2, -1)
    return a * F.silu(b)

def swiglu(a, b):
    return a * F.silu(b)


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
                # keep 8 d² params  ⇒  h ≈ 8/3 ⋅ d (following Shazeer's implementation)
                h  = int(round(8 * d / 3))
                self.fc1 = LinearDebiased(d, 2 * h, bias=False, init_g=G_W_INIT)
                self.fc2 = LinearDebiased(h,     d, bias=True,  init_g=G_W_INIT)
                # bias needs its *own* scalar (2 scales), register a hook
                self.fc2.bias.register_hook(lambda grad: grad * G_B_INIT)
                self.act_fn = lambda u: swiglu(*u.chunk(2, dim=-1))

                #self.fc2.bias.register_hook(lambda g: g.mul_(G_B_INIT))
                self.act_fn = lambda u: swiglu(*u.chunk(2, dim=-1))
            elif (act == "gelu" or act == "relu" or act == "linear"):
                # GELU / ReLU / linear branch  →  4 d² params
                h  = 4 * d
                self.fc1 = LinearDebiased(d, h, bias=False, init_g=G_W_INIT)
                self.fc2 = LinearDebiased(h, d, bias=True,  init_g=G_W_INIT)
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
    """
    Residual MLP with L blocks of the form
    ```
    for k = 1 … L
            h_k =  Wₖ¹ · LN(A_{k-1})                # “pre-act” hidden
            A_k =  A_{k-1} +  Wₖ² · φ(h_k)          # residual update
    ```
    """
    def __init__(self, d, L, ln=True, act="gelu", debias=False):
        super().__init__()
        self.layers = nn.ModuleList([ResidualBlock(d, ln, act, debias) for _ in range(L)])
    def forward(self, x):
        for blk in self.layers: x = blk(x)
        return x

# ----------------- teacher classes and methods ---------------------------------
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
    with matched parameter counts even when act="swiglu". Currently
    the teacher does not use layernorm, however. [TODO - add it?]

    Parameters
    ----------
    input_dim      : dimension of x and y
    teacher_width  : target hidden size for non-SwiGLU activations
    depth          : how many (linear-act-linear) blocks to stack
    act            : one of {"linear","gelu","relu","tanh","swiglu"}
    device         : "cpu" or "cuda"
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


# ===================  OPTIMISER  =================================
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
    return finalize_mx_specs({
        "w_elem_format":      args.elem_format,
        "a_elem_format":      args.elem_format,
        "w_elem_format_bp":   args.elem_format_bp_w,
        "a_elem_format_bp_ex":args.elem_format_bp_ex,
        "a_elem_format_bp_os":args.elem_format_bp_os,
        "block_size": 32,
        "scale_bits": args.scale_bits,
        "custom_cuda": True,
        "quantize_backprop": True if not args.dont_quantize_backprop else False,
        "bfloat": 16,
    })

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

    # whitelist (edit to taste) --------------------------------
    keep_keys = {
        "depth"        : ("L",  "{}"),     # L8
        "width"        : ("D",  "{}"),     # D512
        "batch"        : ("bs", "{}"),     # bs512
        "lr_max"       : ("lr", "{:.0e}"), # lr1e-03
        "elem_format"  : ("fmt","{}"),     # fmtfp8_e4m3
        "scale_bits"   : ("sb", "{}"),     # sb16
        "sgd"          : ("sgd",""),       # flag only when True
        "lr_schedule"  : ("cos",""),       # idem
        "no_ln"        : ("noln",""),      # idem
        "no_act"       : ("lin",""),       # idem
        "act"          : ("act","{}"),     # actgelu (only if not no_act)
        "seed"         : ("s",  "{}"),     # s1337
        "elem_format_bp_w": ("bp","{}"),
    }
    parts = []
    for k in sorted(keep_keys):
        if k not in cfg:          # not present → ignore
            continue
        prefix, fmt = keep_keys[k]
        val = cfg[k]
        if isinstance(val, bool):
            if val:               # only include if True
                parts.append(prefix)
        else:
            parts.append(f"{prefix}{fmt.format(val)}")
    
    # generate random number between 1 and 10000
    random_number = random.randint(1, 1000000)

    folder_name = "_".join(parts) + "_" + str(random_number)
    return out_root / folder_name

# -----------------------------------------------
# hyper‑parameter → short token mapping
# -----------------------------------------------
_KEEP_KEYS = {
    "depth"       : ("L",  "{}"),       # L8
    "width"       : ("D",  "{}"),       # D512
    "batch"       : ("bs", "{}"),       # bs2048
    "lr_max"      : ("lr", "{:.0e}"),   # lr3e-04
    "elem_format" : ("fmt","{}"),       # fmtfp8_e5m2
    "sgd"         : ("sgd",""),         # flag → token if True
    "no_ln"       : ("noln",""),        # …
    "no_act"      : ("lin",""),
    "act"         : ("act","{}"),       # actrelu (only if not no_act)
    "dont_quantize_backprop": ("dqbp",""),
    "seed"        : ("s",  "{}"),       # s1337
    "elem_format_bp_w": ("bp","{}"),   
}


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


# ===================  MAIN  ======================================
def run_dual(args):
    print("Setting seed")
    set_seed(args.seed)
    device = torch.device(args.device)
    args.save_root.mkdir(parents=True, exist_ok=True)

    loss_rec = {"fp32": {}, "mx": {}}
    eps_rec  = {}
    act_rec  = defaultdict(dict)
    group_mean_fp32 = defaultdict(dict)


    #  dirs & wandb
    cfg = vars(args)
    tag, run_dir = build_tags(cfg, random_tag=True, save_root=args.save_root)

    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_tag = f"{tag}_{uuid.uuid4().hex[:4]}"

    wandb.init(
        project=args.wandb_project,
        name   =wandb_tag,
        id     =wandb_tag,
        config =cfg,
        resume ="allow",
        settings=wandb.Settings(init_timeout=120)
    )

    print("Getting model and teacher")
    act_name = "linear" if args.no_act else args.act
    model = ResidualMLP(args.width, args.depth, not args.no_ln, act_name).to(device)
    if args.use_custom_init:
        model.apply(custom_init)
    init_state = {k: v.clone() for k,v in model.state_dict().items()}
    teacher = make_teacher(input_dim=args.width,
                           teacher_width=args.teacher_width,
                           depth=args.teacher_depth,
                           act=args.teacher_act,
                           device="cpu")
    
    teacher_fp32 = copy.deepcopy(teacher).float().to(device)

    print("Making batches")
    batches = [gaussian(args.batch, args.width, device) for _ in range(args.steps_total)]
    val_batches = [gaussian(args.val_batch, args.width, device)
               for _ in range(args.val_steps)]

    g_fp32_mean_store, g_fp32_norm_store = [], []
    param_slices = build_param_slices(model, not args.no_ln)
    global_step=0


    # ========== Phase 1 : FP32 ====================================
    model.load_state_dict(init_state)
    opt32, sched32 = make_optimizer(model, args)
    if args.store_full_gradients:
        g32_store = {}
        g32_lam = {}

    for step,x in enumerate(batches):
        y = teacher_fp32(x) + args.noise_std*torch.randn_like(x, dtype=x.dtype)
        opt32.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        g32 = grad_vec(model, drop_scalars=True).cpu()
        if step % 10 == 0 and args.store_full_gradients:
            g32_store[step] = g32.clone().float() 
        g_fp32_mean_store.append(g32.mean().item())
        g_fp32_norm_store.append(torch.norm(g32).item())
        for name, lo, hi in param_slices:
            group_mean_fp32[step][name] = g32[lo:hi].mean().item() 
        del g32

        opt32.step(); sched32.step() if sched32 else None
        wandb.log({"train/loss_fp32": loss.item(),
                   "lr_fp32": sched32.get_last_lr()[0] if sched32 else args.lr_max}, step=global_step)
        loss_rec["fp32"][step] = loss.item()
        if step % args.val_every == 0:
            v = val_loss(model, teacher_fp32, F.mse_loss, val_batches)
            loss_rec["fp32"][f"val_{step}"] = v
            print(f"val_loss_fp32: {v:.3e}")
            wandb.log({"val/loss_fp32": v}, step=global_step)
        global_step += 1

    
    # ========== Phase 2 : MX ======================================
    print("MX phase")
    if args.debias_MX_gradient:
        model_mx = ResidualMLP(args.width, args.depth,
                            not args.no_ln, act_name,
                            debias=True).to(device)
        # load only the shared parameters; leave the new g scalars
        model_mx.load_state_dict(init_state, strict=False)
    else:
        model_mx = ResidualMLP(args.width, args.depth,
                            not args.no_ln, act_name,
                            debias=False).to(device)
        model_mx.load_state_dict(init_state)          # strict=True OK

    model = model_mx
    opt, sched = make_optimizer(model, args)
    mx_mapping.inject_pyt_ops(make_mx_specs(args))

    for step,x in enumerate(batches):
        y = teacher_fp32(x) + args.noise_std*torch.randn_like(x, dtype=x.dtype)
        opt.zero_grad(set_to_none=True)
        loss_mx = F.mse_loss(model(x), y); loss_mx.backward()

        g_mx = grad_vec(model, drop_scalars=True).cpu()

        if args.debias_MX_gradient:
            for n,p in model.named_parameters():
                if n.endswith(".g"):
                    wandb.log({f"g/{n}": p.item()}, step=global_step)
                if n.endswith(("g_a1","g_a2")):
                    wandb.log({f"g/act_{n}": p.item()}, step=global_step)

        mx_mean = g_mx.mean().item()
        mx_norm = ((g_mx**2).sum().sqrt()).item()

        eps_mean = mx_mean - g_fp32_mean_store[step]
        eps_norm_upper_bound = mx_norm + g_fp32_norm_store[step]
        group_eps_mean = {}

        if step % 10 == 0 and args.store_full_gradients:
            #lam = g32_lam[step]
            #eta_lam = args.lr_max * lam
            eps_vec      = g_mx - g32_store[step]           # ε_t
            eps_norm2    = (eps_vec**2).sum().item()
            g_fp32_norm2 = (g32_store[step]**2).sum().item()
            cos_theta = torch.dot(g_mx, g32_store[step]) / (mx_norm * g_fp32_norm_store[step] + 1e-12)
            sigma2       = eps_norm2 / (g_fp32_norm2 + 1e-12)  # σ_ζ²  (empirical)
            rhs          = 2.0 / (sigma2 + 1e-12)
            zeta_i = eps_vec / g32_store[step]
            sigma_inf = torch.max(torch.abs(zeta_i))
            del g32_store[step] #save some memory
            #rhs = 2.0 / (sigma_inf**2 + 1e-12)

            wandb.log({
                #"hess/lambda_full"      : lam,
                "stab/cos"         : cos_theta,
                #"stab/lhs_eta_lam"      : eta_lam,
                "stab/rhs_2_over_sigma2": rhs,
                #"stab/ratio_lhs_over_rhs": eta_lam / rhs,
                "stab/zeta_norm": sigma2,
                "stab/sigma_inf" : sigma_inf,
                "stab/zeta_i_2_norm": (zeta_i**2).sum().sqrt(),
            }, step=global_step)
            
        for name, lo, hi in param_slices:
            mx_grp_mean  = g_mx[lo:hi].mean().item()
            fp32_grp_mean = group_mean_fp32[step][name]       # cached above
            #fp64_grp_mean = group_mean_fp64[step][name]       # cached above
            group_eps_mean[name] = mx_grp_mean - fp32_grp_mean
            #group_eps_mean_fp64[name] = mx_grp_mean - fp64_grp_mean
            wandb.log({f"eps/mean_{name}": group_eps_mean[name]}, step=global_step)
            #wandb.log({f"eps/mean_fp64_{name}": group_eps_mean_fp64[name]}, step=global_step)

        fp32_norm = g_fp32_norm_store[step]
        #fp64_norm = g_fp64_norm_store[step]


        loss_rec["mx"][step] = loss_mx.item()

        eps_rec[step] = {
            "mean":       eps_mean,
            "norm_upper": eps_norm_upper_bound,
            "mx_norm":    mx_norm,
            "fp32_norm":  fp32_norm,
            "group_mean_fp32": group_eps_mean.copy(),
        }
        
        # activations distribution every val_every
        if step % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                z = x
                acts = []
                hs = []
                for blk_idx, blk in enumerate(model.layers):
                    z = blk.ln(z) if not args.no_ln else z
                    acts.append(z)
                    a1 = blk.fc1(z)
                    h  = blk.act_fn(a1)              # current block activation
                    hs.append(h)
                    W2 = blk.fc2.weight              # (d_out , d)
                    proj = torch.matmul(W2, h.T).abs()   # (d_out , B)
                    numer = proj.mean(dim=1)             # (d_out,)

                    w_norm = (W2 ** 2).sum(dim=1, keepdim=True).sqrt()      # (d_out , 1)
                    h_norm = (h  ** 2).sum(dim=1, keepdim=True).sqrt().T    # (1      , B)

                    denom  = (w_norm * h_norm).mean(dim=1) + 1e-12        
                    align  = (numer / denom.squeeze()).mean()   # scalar

                    wandb.log({f"align/blk{blk_idx}": align.item()}, step=global_step)
                    z = blk.fc2(h) + z               # proceed to next block
                for k,a in enumerate(acts):
                    m, v = a.mean().item(), a.var(unbiased=False).item()
                    act_rec[k][step] = (m, v)
                    wandb.log({f"act/mean_layer{k}": a.mean().item(),
                               f"act/var_layer{k}":  a.var(unbiased=False).item()}, step=global_step)
                for k,h in enumerate(hs):
                    wandb.log({f"act/mean_hid{k}": h.mean().item(),
                               f"act/var_hid{k}":  h.var(unbiased=False).item()}, step=global_step)

                v = val_loss(model, teacher_fp32, F.mse_loss, val_batches)

                print(f"val_loss_mx: {v:.3e}")  
                wandb.log({"val/loss_mx": v}, step=global_step)
            model.train()


        wandb.log({
            "eps_step": step,
            "eps/mean": eps_mean,
            "eps/norm_upper_bound": eps_norm_upper_bound,
            "train/loss_mx": loss_mx.item(),
            "lr": sched.get_last_lr()[0] if sched else args.lr_max,
        }, step=global_step)

        opt.step();  sched.step() if sched else None
        global_step += 1
    wandb.finish()
    payload = {
        "args": vars(args),
        "loss": loss_rec,
        "eps":  eps_rec,
        "activations": act_rec,
    }
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v)
                for k,v in payload.items()},
            run_dir / "summary.pt")
    print("wrote", run_dir / "summary.pt")

# ------------------------- CLI -----------------------------------
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
    p.add_argument("--elem_format", default="fp8_e4m3")
    p.add_argument("--elem_format_bp_w", default="fp8_e4m3")
    p.add_argument("--elem_format_bp_ex", default="fp8_e4m3")
    p.add_argument("--elem_format_bp_os", default="fp8_e4m3")
    p.add_argument("--debias_MX_gradient", action="store_true", help="Debias MX gradients")
    p.add_argument("--use_custom_init", action="store_true", help="use custom init with lower gain")
    p.add_argument("--dont_quantize_backprop", action="store_true")
    p.add_argument("--scale_bits", type=int, default=16)
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
    run_dual(parse())
