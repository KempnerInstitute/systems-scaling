import numpy as np
import pandas as pd
import jax.numpy as jnp
import optax
from jax import jit, vmap

from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt


def chinchilla_curve(params, N, D, masks, vals):
    params = params * masks + vals
    a, b, e, alpha, beta = params
    param_term = jnp.exp(a) / jnp.power(N, alpha)
    data_term = jnp.exp(b) / jnp.power(D, beta)
    entropy_term = jnp.exp(e) * jnp.ones_like(D)
    return param_term + data_term + entropy_term


def kaplan_curve(params, N, D, masks, vals):
    params = params * masks + vals
    a, b, e, alpha, beta = params
    param_term = jnp.power(jnp.exp(a) / N, alpha / beta)
    data_term = jnp.exp(b) / D
    pd_term = jnp.power(param_term + data_term, beta)
    entropy_term = jnp.exp(e) * jnp.ones_like(D)
    return pd_term + entropy_term


def fit(N, D, L, alpha=None, beta=None, E=None, A=None, B=None, kaplan=False):
    if kaplan:
        curve = kaplan_curve
    else:
        curve = chinchilla_curve

    def loss_fn(params, N, D, L, masks, vals):
        preds = curve(params, N, D, masks, vals)
        return jnp.mean(optax.losses.huber_loss(jnp.log(preds), jnp.log(L), delta=1e-3))
        # return jnp.mean(optax.losses.huber_loss(preds, L, delta=1e-3))

    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    optimizer = optax.lbfgs()

    @jit
    def update(params, opt_state, N, D, L, masks, vals):
        value, grad = value_and_grad(params, N, D, L, masks, vals, state=opt_state)
        updates, opt_state = optimizer.update(
            grad,
            opt_state,
            params,
            value=value,
            grad=grad,
            value_fn=loss_fn,
            N=N,
            D=D,
            L=L,
            masks=masks,
            vals=vals,
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def run(params, N, D, L, masks, vals):
        opt_state = optimizer.init(params)

        for _ in range(200):
            params, opt_state = update(params, opt_state, N, D, L, masks, vals)
        current_loss = loss_fn(params, N, D, L, masks, vals)
        return params, current_loss

    run_all = vmap(run, in_axes=(0, None, None, None, None, None), out_axes=(0, 0))

    masks = jnp.array([int(x is None) for x in [alpha, beta, E, A, B]])
    vals = jnp.array([x or 0 for x in [alpha, beta, E, A, B]])

    grids = [
        jnp.arange(5, 25, 5),  # log a
        jnp.arange(5, 25, 5),  # log b
        jnp.arange(0.5, 1, 0.5),  # log e
        jnp.arange(0.4, 0.5, 0.2),  # jnp.arange(0.2, 0.8, 0.2),  # alpha
        jnp.arange(0.4, 0.5, 0.2),  # jnp.arange(0.2, 0.8, 0.2),  # beta
    ]

    inits = jnp.array(jnp.meshgrid(*grids)).reshape((5, -1)).T

    opt_params, opt_loss = run_all(inits, N, D, L, masks, vals)
    rsquared = []
    for i in range(inits.shape[0]):
        residual = curve(opt_params[i], N, D, masks, vals) - L
        r2 = 1 - jnp.sum(jnp.power(residual, 2)) / jnp.sum(
            jnp.power(L - jnp.mean(L), 2)
        )
        rsquared.append(r2)

    opt_loss = jnp.array(opt_loss)
    opt_params = jnp.array(opt_params) * masks + vals
    min_index = np.argsort(opt_loss)
    loss = opt_loss[min_index]
    params = opt_params[min_index]
    r_squared = jnp.array(rsquared)[min_index]

    sol_df = pd.DataFrame(
        {
            "A": np.exp(params[:, 0]),
            "B": np.exp(params[:, 1]),
            "E": np.exp(params[:, 2]),
            "alpha": params[:, 3],
            "beta": params[:, 4],
            "loss": loss.flatten(),
            "r_squared": r_squared.flatten(),
        }
    )
    opt_df = sol_df.iloc[sol_df["loss"].idxmin()]
    return opt_df, sol_df


def get_params(params):
    A = params["A"]
    B = params["B"]
    E = params["E"]
    alpha = params["alpha"]
    beta = params["beta"]
    return A, B, E, alpha, beta


def plot_contours(
    ax, N, D, L, opt_params, masks, vals, extrapolation, kaplan=False, ncontour=40
):
    A, B, E, alpha, beta = get_params(opt_params)
    print(f"A = {A}, B = {B}, E = {E}, alpha = {alpha}, beta = {beta}")
    F = 6 * np.array(N, dtype=np.float64) * np.array(D, dtype=np.float64)

    # Optimal N line for chinchilla
    if not kaplan:
        G = np.power((alpha * A) / (beta * B), 1 / (alpha + beta))
    else:
        G = alpha * np.power(A, alpha / beta) / (beta * B)
    a = beta / (alpha + beta)
    b = alpha / (alpha + beta)
    print(f"G = {G}, a = {a}, b = {b}")
    x = np.linspace(min(F) / 2, max(F) * 25, 100)
    if not kaplan:
        y = G * np.power(x / 6, a)
    else:
        y = np.power(G * x / 6, a)
    ax.plot(x, y, c="red", linestyle="--", label="Optimal N")

    # Contours
    # x = np.geomspace(np.min(F) / 2, np.max(F) * 25, 500)
    # y = np.geomspace(np.min(N) / 2, np.max(N) * 3, 500)
    x = np.geomspace(1e17, 2e21, 500)
    y = np.geomspace(1e7, 5e9, 500)
    X, Y = np.meshgrid(x, y)
    if kaplan:
        curve = kaplan_curve
    else:
        curve = chinchilla_curve
    params = np.array([np.log(A), np.log(B), np.log(E), alpha, beta])
    Z = curve(params, Y, X / (6 * Y), masks, vals)

    cmap = cm.viridis
    # extrapolation
    if extrapolation is not None:
        x, y, z = extrapolation  # C, N, L

        norm = Normalize(vmin=min(L.min(), z), vmax=L.max())

        scatter = ax.scatter(x, y, c=z, cmap=cmap, norm=norm, s=20, marker="*")
    else:
        norm = Normalize(vmin=L.min(), vmax=L.max())

    scatter = ax.scatter(F, N, c=L, cmap=cmap, norm=norm, s=20)
    contour = ax.contour(X, Y, Z, levels=ncontour, cmap=cmap, norm=norm)
    plt.colorbar(scatter, label="Train Loss")

    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Parameters")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylim(1e7, 5e9)
    ax.set_xlim(1e17, 2e21)
    # ax.set_ylim(np.min(N) / 2, np.max(N) * 3)
    # ax.set_xlim(np.min(F) / 2, np.max(F) * 25)
    return scatter, contour, cmap, norm
