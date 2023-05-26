import jax
import jax.numpy as jnp
import flax.linen as nn
from torchvision.datasets import MNIST

from loguru import logger
from functools import partial
from clu import metrics
from flax.training import train_state
from flax import struct
import optax

from .helpers.jax import *
from .model.jax.model import Unet


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, rng, learning_rate):
    params = module.init(rng, jnp.ones([1, 14, 14, 1]), jnp.ones([1, 1]))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty())


@jax.jit # @partial(jax.jit, static_argnums=(0,))
def train_step(state, noised_sample, steps, noise):
    def loss_fn(params):
        predicted_noise = state.apply_fn({'params': params}, noised_sample, steps)
        loss = optax.l2_loss(predictions=predicted_noise, targets=noise).mean()
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def validate(state, noised_sample, steps, noise):
    predicted_noise = state.apply_fn({'params': state.params}, noised_sample, steps)
    loss = optax.l2_loss(predictions=predicted_noise, targets=noise).mean()
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


def sampling(state, noised_sample, max_steps):
    steps = jnp.arange(1, max_steps+1).astype('int32').reshape(-1, 1)
    steps = jnp.repeat(steps, noised_sample.shape[0], axis=1)
    betas = linear_beta_schedule(max_steps)
    alphas = 1 - betas
    alphas_overline = alphas_cum_prod(betas, steps[:, 0])

    C1 = 1 / jnp.sqrt(alphas)
    C2 = (1 - alphas) / jnp.sqrt(1 - alphas_overline)
    for i, t in enumerate(steps):
        if i > 0:
            z = sample_noise(noised_sample)
        else:
            z = jnp.zeros_like(noised_sample)
        time = t[:, None]
        predicted_noise = state.apply_fn({'params': state.params}, noised_sample, time)
        noised_sample = C1[i] * (noised_sample - C2[i] * predicted_noise) + z * betas[i]

    return noised_sample


def experiment(model: nn.Module, data: jnp.array, num_epoches: int = 5, batch_size: int = 64, max_steps: int = 300, lr: float = 1e-3):
    rng = jax.random.PRNGKey(0)
    state = create_train_state(model, rng, lr)
    loss_history = []

    for epoch in range(num_epoches):
        logger.info(f'Start {epoch} epoch')
        for i, real_data in enumerate(sampling_real(data, batch_size)):

            steps = sample_steps(real_data, max_steps, rng)[:, None]
            noise = sample_noise(real_data, max_steps, rng)[:, :, :, None]
            real_data = real_data[:, :, :, None]
            noised_sample = noised_data(real_data, steps, noise, max_steps)
            state = train_step(state, noised_sample, steps, noise)
            state = validate(state, noised_sample, steps, noise)

            if i % 1000:
                val_noise = sample_noise(real_data, max_steps, rng)
                generated = sampling(state, val_noise, max_steps)
                plot_samples(generated, n_cols=8, path='./')
                logger.info(f"Loss: %.4f" % float(state.metrics.compute().get('loss')))
        loss_history.append(float(state.metrics.compute().get('loss')))

    val_noise = sample_noise(real_data, max_steps, rng)
    generated = sampling(state, val_noise, max_steps)

    return state, loss_history, generated


if __name__ == '__main__':

    logger.info('Import data')
    training = MNIST(
        root="data/",
        train=True,
        download=True,
    ).data.numpy()

    test = MNIST(
        root="data/",
        train=False,
        download=True,
    ).data.numpy()

    logger.info('Prepare data')
    training = preproc_data(training)
    test = preproc_data(test)

    logger.info('Init experiment')
    model = Unet()
    state, loss, samples = experiment(model, training)
