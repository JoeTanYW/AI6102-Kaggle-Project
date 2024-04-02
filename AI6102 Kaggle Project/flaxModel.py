import jax as jax
import jax.numpy as jnp
from flax import linen as nn

from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax                           # Common loss functions and optimizers

class ShirtNN(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        # x = nn.sigmoid(x)
        return x
    

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss') # type: ignore


class TrainState(train_state.TrainState):
  metrics: Metrics


def create_train_state(module, rng, learning_rate, momentum):
  """Creates an initial `TrainState`."""
  params = module.init(rng, jnp.ones([1, 26]))[
      'params']  # initialize parameters by passing a template image
  tx = optax.sgd(learning_rate, momentum)
  return TrainState.create(
      apply_fn=module.apply, params=params, tx=tx,
      metrics=Metrics.empty())


@jax.jit
def train_step(state, x, y):
  """Train for a single step."""
  def loss_fn(params):
    logit = state.apply_fn({'params': params}, x)
    loss = optax.sigmoid_binary_cross_entropy(
        logit, y).mean()
    return loss
  grad_fn = jax.grad(loss_fn)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state


@jax.jit
def pred_step(state, x):
  logits = state.apply_fn({'params': state.params}, x)
  return jnp.round(jax.nn.sigmoid(logits))


@jax.jit
def compute_metrics(*, state, x, y):
  logit = state.apply_fn({'params': state.params}, x)
  loss = optax.sigmoid_binary_cross_entropy(
      logits=logit, labels=y).mean()
  metric_updates = state.metrics.single_from_model_output(
      logits=logit, labels=y, loss=loss)
  metrics = state.metrics.merge(metric_updates)
  state = state.replace(metrics=metrics)
  return state


def init_model(seed=0, learning_rate = 0.01, momentum = 0.9):
    shirt = ShirtNN()
    init_rng = jax.random.key(seed)
    return create_train_state(shirt, init_rng, learning_rate, momentum)


def train(state, x, y, num_epochs=50):
    metrics_history = {'train_loss': [],
                       'train_accuracy': []}
    for ep in range(num_epochs):
        state = train_step(state, x, y)
        state = compute_metrics(state=state, x=x, y=y)
        for metric, value in state.metrics.compute().items():
            metrics_history[f'train_{metric}'].append(value)
        state = state.replace(metrics=state.metrics.empty())
        print(
            f"{ep + 1} | loss: {metrics_history['train_loss'][-1]}; accuracy: {metrics_history['train_accuracy'][-1] * 100}")
    return state


def f1(state, x, targets):
    preds = pred_step(state, x)
    true_positives = jnp.sum(preds * targets)
    precision = true_positives / jnp.sum(preds)
    recall = true_positives / jnp.sum(targets)
    return 2 * precision * recall / (precision + recall)


def evaluate(state, x_train, y_train, x_val, y_val):
    state = train(state, jnp.asarray(x_train), jnp.asarray(y_train))
    return f1(state, x_val, y_val)
