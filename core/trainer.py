import jax
from flax.training import train_state
import optax
import jax.numpy as jnp
import numpy as np

from core.model import ToySuperpositionModel


# Define the Data Generator
def generate_sparse_batch(key, batch_size, num_features, sparsity):
    """
    Generates synthetic data where features are mostly zero.
    Requires an explicit JAX PRNGKey.
    """
    key_mag, key_mask = jax.random.split(key)

    # Generate uniform magnitudes between 0 and 1
    magnitudes = jax.random.uniform(key_mag, shape=(batch_size, num_features))

    # Create a mask where features are present based on (1 - sparsity)
    mask = (
        jax.random.uniform(key_mask, shape=(batch_size, num_features)) > sparsity
    ).astype(jnp.float32)

    return magnitudes * mask


# Define the Training Step
# We use jax.jit to compile this function into highly optimized XLA code
@jax.jit
def train_step(state, x, importance):
    def loss_fn(params):
        # Forward pass
        x_prime = state.apply_fn({"params": params}, x)

        # Calculate Weighted Mean Squared Error
        squared_error = (x - x_prime) ** 2
        weighted_error = squared_error * importance
        loss = jnp.mean(jnp.sum(weighted_error, axis=-1))
        return loss

    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    # Update model state using the optimizer
    state = state.apply_gradients(grads=grads)

    return state, loss


# Define the Training Loop
def train_toy_model(
    num_features=20, hidden_dim=5, sparsity=0.99, steps=5000, batch_size=1024
):
    # Initialize random key
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)

    # Initialize model and parameters
    model = ToySuperpositionModel(num_features=num_features, hidden_dim=hidden_dim)
    dummy_input = jnp.ones((batch_size, num_features))
    params = model.init(init_key, dummy_input)["params"]

    # Initialize Optax optimizer
    tx = optax.adamw(learning_rate=0.001)

    # Create the TrainState (holds params, optimizer state, and apply_fn)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Create feature importance weights (Exponential decay)
    importance = jnp.array([0.9**i for i in range(num_features)])

    # Training Loop
    for step in range(steps):
        # We need a new random key for every batch
        key, batch_key = jax.random.split(key)

        # 1. Get batch of sparse data
        x = generate_sparse_batch(batch_key, batch_size, num_features, sparsity)

        # 2. Perform one step of training
        state, loss = train_step(state, x, importance)

        if step % 1000 == 0:
            print(f"Step {step} | Loss: {loss:.4f}")

    return state, model


def train_toy_model_with_history(
    num_features=20, hidden_dim=5, sparsity=0.99, steps=5000, batch_size=1024
):
    """Trains the model and saves the weight matrix at specific steps."""
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)

    model = ToySuperpositionModel(num_features=num_features, hidden_dim=hidden_dim)
    dummy_input = jnp.ones((batch_size, num_features))
    params = model.init(init_key, dummy_input)["params"]

    tx = optax.adamw(
        learning_rate=0.002
    )  # Slightly higher LR to speed up geometry formation

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    importance = jnp.array([0.9**i for i in range(num_features)])

    # We will save the weights at these specific steps to see the progression
    save_steps = [0, 200, 1000, 2500, steps - 1]
    history = []

    for step in range(steps):
        key, batch_key = jax.random.split(key)
        x = generate_sparse_batch(batch_key, batch_size, num_features, sparsity)

        state, loss = train_step(state, x, importance)

        # Save the current state of the weights W
        if step in save_steps:
            # Convert to standard numpy array immediately to store it
            current_W = np.asarray(state.params["W"])
            history.append((step, current_W))

        if step % 1000 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    return state, history
