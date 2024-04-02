import jax as jax #it doesn't work otherwise 
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"


def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]





def relu(x):
  return jnp.maximum(0, x)


def predict(params, inpt):
  # per-example predictions
  activations = inpt
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)

  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return jax.nn.sigmoid(logits)  # - logsumexp(logits)


batched_predict = vmap(predict, in_axes=(None, 0))


def accuracy(params, vecs, targets):
  target_class = targets
  predicted_class = jnp.round(batched_predict(params, vecs))
  return jnp.mean(predicted_class == target_class)


def f1(params, vecs, targets):
  preds = jnp.round(batched_predict(params, vecs))
  true_positives = jnp.sum(preds * targets)
  precision = true_positives / jnp.sum(preds)
  recall = true_positives / jnp.sum(targets)
  return 2 * precision * recall / (precision + recall)


def loss(params, images, targets):
  preds = batched_predict(params, images)
  return -jnp.mean(preds * targets + (1 - preds) * (1 - targets))


@jit
def update(params, x, y, step_size=0.01):
  grads = grad(loss)(params, x, y)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]


def get_train_batches(x, y):
  return [(x, y)]  # TODO: can I do better?


def train(params, x, y, num_epochs):
  for _ in range(num_epochs):
    for x, y in get_train_batches(x, y):
      params = update(params, x, y)
  return params


def evaluate(params, train_x, train_y, test_x, test_y, num_epochs=50):
  params = train(params, train_x, train_y, num_epochs)
  return f1(params, test_x, test_y)


def get_model(layer_sizes=[26, 64, 32, 16, 1]): return init_network_params(
    layer_sizes, random.key(0))

