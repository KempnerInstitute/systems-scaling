import numpy as np
import pandas as pd

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap

from scipy.optimize import curve_fit


display_dict = {
    "fineweb-100b": "FineWeb",
    "fineweb-edu-100b": "FineWeb-Edu",
    "smollm-corpus": "SmolLM Corpus",
    "smollm": "SmolLM Corpus",
    "proof-pile-2": "ProofPile 2",
    "starcoder": "StarCoder",
    "slimpajama-chunk1": "SlimPajama",
    "slimpajama": "SlimPajama",
}

val_map = {
    "smollm-corpus": "eval/smollm_val/CrossEntropyLoss",
    "fineweb-edu-100b": "eval/fineweb_edu_100b_val/CrossEntropyLoss",
    "slimpajama-chunk1": "eval/slimpajama_val/CrossEntropyLoss",
    "fineweb-100b": "eval/fineweb_100b_val/CrossEntropyLoss",
    "proof-pile-2": "eval/proof_pile_2_val/CrossEntropyLoss",
    "starcoder": "eval/starcoder_val/CrossEntropyLoss",
}

val_display_dict = {v: display_dict[k] for k, v in val_map.items()}
val_display_dict["mmlu_suite_ce_loss"] = "MMLU"
val_display_dict["eval/downstream_ce_loss/hellaswag_test_ce_loss"] = "Hellaswag"
val_display_dict["eval/downstream_ce_loss/arc_easy_test_ce_loss"] = "ARC-Easy"
val_display_dict["eval/downstream_ce_loss/arc_challenge_test_ce_loss"] = "ARC-Challenge"
val_display_dict["eval/downstream_ce_loss/mmlu_humanities_test_ce_loss"] = (
    "MMLU-Humanities"
)
val_display_dict["eval/downstream_ce_loss/openbook_qa_test_ce_loss"] = "OpenBookQA"
val_display_dict["eval/downstream_ce_loss/piqa_test_ce_loss"] = "PIQA"
val_display_dict["eval/downstream_ce_loss/sciq_test_ce_loss"] = "SciQ"
val_display_dict["eval/downstream_ce_loss/winogrande_test_ce_loss"] = "Winogrande"
val_display_dict["eval/downstream_ce_loss/mmlu_stem_test_ce_loss"] = "MMLU-STEM"
val_display_dict["eval/downstream_ce_loss/mmlu_other_test_ce_loss"] = "MMLU-Other"
val_display_dict["eval/downstream_ce_loss/mmlu_social_sciences_test_ce_loss"] = (
    "MMLU-Social Sciences"
)


def plot_with_line(
    ax, x, y, color, label, logx=False, logy=False, extrapolate=False, add_x=None
):
    ax.scatter(x, y, color=color, label=label, alpha=0.5, s=10)
    # sort
    x, y = zip(*sorted(zip(x, y)))
    original_y = y

    if logx:
        x = np.log(x)
    if logy:
        y = np.log(y)
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)

    if add_x is not None:
        if logx:
            add_x = np.log(add_x)
        x = np.concatenate([[add_x], x])

    y = poly(x)
    if logx:
        x = np.exp(x)
    if logy:
        y = np.exp(y)
    ax.plot(x, y, color=color)

    if add_x is not None:
        y = y[1:]
    r_squared = 1 - (
        np.sum((original_y - y) ** 2) / np.sum((original_y - np.mean(original_y)) ** 2)
    )

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    results_df = pd.DataFrame({"params": [coeffs], "r_squared": r_squared})

    return results_df


def power_law_with_constant(x, a, b, c, d):
    xc = np.maximum(x - c, 0)
    xc = np.where(xc == 0, np.nan, xc)
    return b * np.power(xc, a) + d


def get_simple_power_law(c):
    def power_law_with_constant(x, a, b, d):
        return b * np.power(x - c, a) + d

    return power_law_with_constant


def solve(bx, by, x, y):
    X = jnp.stack([jnp.log(x - bx), jnp.ones_like(x)], axis=1)
    params, loss, _, _ = jnp.linalg.lstsq(X, jnp.log(y - by), rcond=0)
    # select by real l2 loss
    preds = jnp.exp(jnp.dot(X, params)) + by
    loss = jnp.square(preds - y).mean()
    return params, loss.mean()


solve_all = vmap(solve, in_axes=(0, 0, None, None), out_axes=(0, 0))


def fit_power_law(x, y, ex=None, ey=None):

    if ex is not None and ey is None:
        partial_power_law = get_simple_power_law(ex)
        params, covariance = curve_fit(
            partial_power_law,
            x,
            y,
            maxfev=500000,
            p0=[1, 1, 0],
            bounds=([0, 0, 0], [np.inf, np.inf, np.min(y)]),
            method="trf",
        )
        params = np.array([params[0], params[1], ex, params[2]])

    else:
        x = jnp.array(x, dtype=jnp.float64)
        y = jnp.array(y, dtype=jnp.float64)
        if ex is not None:
            bx = jnp.array([ex], dtype=jnp.float64)
        else:
            bx = jnp.linspace(0.0, jnp.min(x) * 1.0, 100, dtype=jnp.float64)
        if ey is not None:
            by = jnp.array([ey], dtype=jnp.float64)
        else:
            by = jnp.linspace(0.0, jnp.min(y), 100, dtype=jnp.float64)

        grid = jnp.meshgrid(bx, by, indexing="ij")
        bx = grid[0].flatten()
        by = grid[1].flatten()

        params, loss = solve_all(bx, by, x, y)
        # overwrite nans
        loss = jnp.where(jnp.isnan(loss), jnp.inf, loss)
        best_idx = jnp.argmin(loss)
        bx, by = bx[best_idx], by[best_idx]
        bs = jnp.array([bx, by])
        params = jnp.concatenate([params[best_idx], bs])
        params = np.array(params)
        params[1] = np.exp(params[1])  # Fix log scale
    return params


def plot_with_power_law(
    ax, x, y, color, label, add_x=None, ex=None, ey=None, fit=True, geom=False
):
    ax.scatter(x, y, color=color, label=label, alpha=0.5, s=10)

    params = fit_power_law(x, y, ex, ey)

    preds = power_law_with_constant(x, *params)
    r_squared = 1 - (np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2))

    # x = np.sort(x)
    if geom:
        x = np.geomspace(np.min(x), np.max(x), 100)
    else:
        x = np.linspace(np.min(x), np.max(x), 100)

    if add_x is not None:
        if add_x < np.min(x):
            if geom:
                x1 = np.geomspace(add_x, np.min(x), 100)
            else:
                x1 = np.linspace(add_x, np.min(x), 100)
        else:
            if geom:
                x1 = np.geomspace(np.max(x), add_x, 100)
            else:
                x1 = np.linspace(np.max(x), add_x, 100)
    y = power_law_with_constant(x, *params)
    if fit:
        ax.plot(x, y, color=color)
        if add_x is not None:
            y1 = power_law_with_constant(x1, *params)
            ax.plot(x1, y1, color=color, linestyle="--")

    results_df = pd.DataFrame({"params": [params], "r_squared": r_squared})
    return results_df


def soft_min(a, b, alpha=10):
    return -np.log(np.exp(-alpha * a) + np.exp(-alpha * b)) / alpha


def piecewise_power_law_with_constant(x, a, b, c, d, e):
    return soft_min(e, b * np.power(x - c, a) + d)


def get_simple_power_law_piecewise(c):
    def power_law_with_constant(x, a, b, d, e):
        return soft_min(e, b * np.power(x - c, a) + d)

    return power_law_with_constant


def fit_piecewise_power_law(x, y, ex, ey=None):

    partial_power_law = get_simple_power_law_piecewise(ex)
    params, covariance = curve_fit(
        partial_power_law,
        x,
        y,
        maxfev=500000,
        p0=[1, 1, 0, 1],
        bounds=([0, 0, 0, 0], [np.inf, np.inf, np.min(y), 1]),
        method="trf",
    )
    params = np.array([params[0], params[1], ex, params[2], params[3]])
    return params


def plot_with_piecewise_power_law(
    ax, x, y, color, label, ex, add_x=None, ey=None, fit=True
):
    ax.scatter(x, y, color=color, label=label, alpha=0.5, s=10)

    params = fit_piecewise_power_law(x, y, ex, ey)

    preds = piecewise_power_law_with_constant(x, *params)
    r_squared = 1 - (np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2))

    x = np.sort(x)
    if add_x is not None:
        x1 = np.linspace(add_x, np.min(x), 100)
        # x = np.concatenate([[add_x], x])
    y = piecewise_power_law_with_constant(x, *params)
    if fit:
        ax.plot(x, y, color=color)
        if add_x is not None:
            y1 = piecewise_power_law_with_constant(x1, *params)
            ax.plot(x1, y1, color=color, linestyle="--")

    results_df = pd.DataFrame({"params": [params], "r_squared": r_squared})
    return results_df
