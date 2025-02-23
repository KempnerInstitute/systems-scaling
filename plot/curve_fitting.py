import pandas as pd
import numpy as np

from utils_plotting import val_map
from utils_scaling import fit
from utils_scaling import chinchilla_curve, kaplan_curve

import jax.numpy as jnp

df = pd.read_csv("data/sweep.csv")
df_big = pd.read_csv("data/extrapolation.csv")

def get_data(drop_df, key = "train/CrossEntropyLoss"):
    drop_df = drop_df.dropna(subset=[key])

    N = jnp.array(drop_df["params"], dtype=jnp.float32)
    D = jnp.array(drop_df["tokens"], dtype=jnp.float32)
    L = jnp.array(drop_df[key], dtype=jnp.float32)
    return N, D, L

def get_params(params):
    A = params["A"]
    B = params["B"]
    E = params["E"]
    alpha = params["alpha"]
    beta = params["beta"]
    return A, B, E, alpha, beta


losses = [col for col in df.columns if ("loss" in col or "CrossEntropyLoss" in col) and 
                          "ctx" not in col and "5shot" not in col and "imbue" not in col and "train" not in col]

opt_params_df = pd.DataFrame()
for kaplan in [False, True]:
    for i, data in enumerate(val_map.keys()):
        for loss_name in losses:
            print(f"Data = {data}, Loss = {loss_name}")

            drop_df = df[df["data"] == data]

            N, D, L = get_data(drop_df, loss_name)

            params, _ = fit(N, D, L, kaplan=kaplan)

            params["kaplan"] = kaplan
            params["data"] = data
            params["loss_name"] = loss_name
            params = pd.DataFrame([params.values], columns=params.index, index = [i])
            opt_params_df = pd.concat([opt_params_df, params])


# add columns for extrapolations and predictions

data = list(df_big["data"])
N = df_big["params"]
D = df_big["tokens"]    

extrap_preds = []
extrap_losses = []
for i, row in opt_params_df.iterrows():
    data_idx = data.index(row["data"])
    n = np.array(N[data_idx], dtype=np.float64)
    d = np.array(D[data_idx], dtype=np.float64)
    A, B, E, alpha, beta = get_params(row)
    params = np.array([np.log(A), np.log(B), np.log(E), alpha, beta])
    if row["kaplan"]:
        extrap_preds.append(kaplan_curve(params, n, d, np.ones(5), np.zeros(5)))
    else:
        extrap_preds.append(chinchilla_curve(params, n, d, np.ones(5), np.zeros(5)))
    extrap_losses.append(df_big[row["loss_name"]][data_idx])
    print(f"Data = {row['data']}, Loss = {row['loss_name']}, Prediction = {extrap_preds[-1]}, Actual = {extrap_losses[-1]}")

opt_params_df["extrap_pred"] = extrap_preds
opt_params_df["extrap_loss"] = extrap_losses

opt_params_df.to_csv("data/opt_params_data.csv", index=False)


print("Finished!!")