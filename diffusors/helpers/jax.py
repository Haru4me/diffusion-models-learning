import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from pathlib import Path
from functools import partial


def preproc_data(data: np.ndarray) -> np.ndarray:
    data = jnp.array(data)
    data = data.astype(float)
    data /= 255
    return jax.image.resize(data, shape=(data.shape[0], 14, 14), method="bicubic", )


@partial(jax.jit, static_argnums=(1, 2, 3))
def sampling_real(data: np.ndarray, batch_size: int = 32, shuffle: bool = True, rng = None) -> jnp.array:
    if rng == None:
        seed = random.randrange(sys.maxsize)
        rng = jax.random.PRNGKey(seed)
    indexes = jnp.arange(data.shape[0]).astype(int)
    if shuffle:
        indexes = jax.random.permutation(rng, indexes)
    num_batches = data.shape[0] // batch_size
    batches_indexes = jnp.reshape(indexes[:num_batches * batch_size], (num_batches, batch_size))
    return data[batches_indexes]


@jax.jit
def sample_steps(data: jnp.array, max_steps: int = 100, rng = None) -> jnp.array:
    if rng == None:
        seed = random.randrange(sys.maxsize)
        rng = jax.random.PRNGKey(seed)
    batch_size = data.shape[0]
    return jax.random.uniform(rng, minval=0, maxval=max_steps, shape=(batch_size, )).round().astype(int)


@jax.jit
def sample_noise(data: jnp.array, batch_size: int = 32, rng = None) -> jnp.array:
    if rng == None:
        seed = random.randrange(sys.maxsize)
        rng = jax.random.PRNGKey(seed)
    return jax.random.normal(rng, shape=data.shape)


@partial(jax.jit, static_argnums=(0,))
def linear_beta_schedule(max_steps: int = 100):
    beta_start = 0.0001
    beta_end = 0.02
    return jnp.linspace(beta_start, beta_end, max_steps)


@jax.jit
def alphas_cum_prod(betas, steps):
    alphas = 1 - betas
    alphas_cum_prod = alphas.cumprod()
    return alphas_cum_prod[steps]


@jax.jit
def noised_data(data: jnp.array, steps: jnp.array, noise: jnp.array) -> jnp.array:

    betas = linear_beta_schedule()
    alphas = alphas_cum_prod(betas, steps)
    alpha = alphas[:, None, None]

    biase = jnp.multiply(data, jnp.sqrt(alphas)[:, None, None])
    variance = jnp.multiply(noise, jnp.sqrt(1 - alphas)[:, None, None])

    return  biase + variance


def plot_samples(samples: jnp.array, n_cols: int = None, n_rows: int = None, path: str = None):
    rows = []
    if n_cols is not None:
        n_rows = samples.shape[0] // n_cols
    elif n_rows is not None:
        n_cols = samples.shape[0] // n_rows
    else:
        raise ValueError("Choose cols or rows")
    for i in range(n_rows):
        start = n_cols * i
        end = n_cols * (i + 1)
        rows.append(jnp.hstack(samples[start:end]))
    grid = jnp.vstack(rows)
    plt.imshow(grid, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(Path(path) / 'sample.png')
