import jax.numpy as jnp
from jax import jit, vmap
from jax.random import PRNGKey, split, normal
import pickle
import jax


available_devices = jax.devices()
print(available_devices)


# parameter choices ---------
Ds = jnp.arange(1000, 2000, 30)
keys = jnp.arange(Ds.shape[0])
M = 1200000
betas = [0.3, 0.4, 0.7, 1.0]
sigma_w = 1
sigma_v = 1
sigma_eps = 0
gamma = 0
num_samples = {0.2: 2000, 0.3: 2000, 0.4: 2000, 0.7: 2000, 1.: 2000} # smaller beta will generally require more samples due to poor conditioning
#----------------------------

def generate_dataset(D, M, beta, sigma, key):
    i_indices = jnp.arange(1, M + 1)
    lambda_i = (1 / i_indices) ** (1 + beta) 
    sqrt_lambda_i = jnp.sqrt(lambda_i)

    key_1, key_2 = split(key)
    x = sqrt_lambda_i * normal(key_1, (D, M), dtype=jnp.float16)
    return x, None

def generate_target(N, M, sigma_w, sigma_v, key):
    key_1, key_2 = split(key)
    w = normal(key_1, (M, 1), dtype=jnp.float16) * sigma_w
    v = normal(key_2, (N, M), dtype=jnp.float16) * sigma_v / jnp.sqrt(M)
    return w, v

def test_loss(x, w, v, gamma, val_x):
    phi = x @ v.T
    wTx = x @ w
    A = phi.T @ phi + gamma * jnp.eye(v.shape[0])
    q = jnp.linalg.inv(A)
    theta = wTx.T @ phi @ q
    val_phi = val_x @ v.T
    y_pred = theta @ val_phi.T
    y_val = w.T @ val_x.T
    test_loss = 0.5 * jnp.mean((y_pred - y_val) ** 2)
    return test_loss

test_loss = jax.jit(test_loss)

def run_experiment(N):
    losses, losses_averaged = {}, {}
    for beta in betas:
        losses[N], losses_averaged[N] = {}, {}
        for i in range(num_samples[beta]):
            if i%100 == 0:
                print(f'N: {N}, beta: {beta}, sample: {i}')
            outer_key = PRNGKey(i)
            key_1, key_2 = split(outer_key)
            val_x, _ = generate_dataset(Ds[-1], M, beta, sigma_eps, key_1)
            w, v = generate_target(N, M, sigma_w, sigma_v, key_2)
            inner_losses = []
            for D, key in zip(Ds, keys):
                key = PRNGKey(key + num_samples[beta])
                x, _ = generate_dataset(D, M, beta, 0, key)
                loss = test_loss(x, w, v, gamma, val_x)
                if i==0:
                    print("test_loss is computed on device:", loss.device)
                inner_losses.append(loss)
            if beta not in losses[N]:
                losses[N][beta] = []
            else:
                losses[N][beta].append(jnp.array(inner_losses))
        losses_averaged[N][beta] = jnp.array(losses[N][beta]).mean(axis=0)
        with open(f'losses_N_{N}_beta_{beta}.pkl', 'wb') as f:
            pickle.dump(losses, f)
        with open(f'losses_averaged_N_{N}_beta_{beta}.pkl', 'wb') as f:
            pickle.dump(losses_averaged, f)

if __name__ == "__main__":
    import sys
    N = int(sys.argv[1])
    run_experiment(N)