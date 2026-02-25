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


def generate_correlated_batch(key, batch_size, num_features, sparsity, group_size=2):
    """
    Generates synthetic data where features co-occur in groups.
    E.g., if group_size=2, feature 0 and 1 always activate together, 2 and 3 activate together, etc.
    """
    key_mag, key_mask = jax.random.split(key)

    # Magnitudes are still independent for every feature
    magnitudes = jax.random.uniform(key_mag, shape=(batch_size, num_features))

    # Sparsity mask is generated per GROUP, not per feature
    num_groups = num_features // group_size
    group_mask = (
        jax.random.uniform(key_mask, shape=(batch_size, num_groups)) > sparsity
    ).astype(jnp.float32)

    # Repeat the mask so all features in a group share the exact same 0 or 1
    # Example: group_mask [1, 0] -> expanded mask [1, 1, 0, 0]
    mask = jnp.repeat(group_mask, group_size, axis=1)

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


def train_correlated_toy_model(
    num_features=4,
    hidden_dim=2,
    sparsity=0.7,
    steps=8000,
    batch_size=1024,
    group_size=2,
):
    key = jax.random.PRNGKey(123)
    key, init_key = jax.random.split(key)

    # 1. Reuse your existing ToySuperpositionModel
    model = ToySuperpositionModel(num_features=num_features, hidden_dim=hidden_dim)
    dummy_input = jnp.ones((batch_size, num_features))
    params = model.init(init_key, dummy_input)["params"]

    # We use a slightly higher learning rate to ensure it settles into the geometric shape
    tx = optax.adamw(learning_rate=0.005)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Use uniform importance (all 1.0) so the model treats all features equally
    importance = jnp.ones(num_features)

    for step in range(steps):
        key, batch_key = jax.random.split(key)

        # 2. Use the NEW correlated data generator
        x = generate_correlated_batch(
            batch_key, batch_size, num_features, sparsity, group_size
        )

        # 3. Reuse your existing train_step
        state, loss = train_step(state, x, importance)

        if step % 2000 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    return state


def train_correlated_toy_model_with_history(
    num_features=20,
    hidden_dim=5,
    sparsity=0.9,
    steps=5000,
    batch_size=1024,
    group_size=2,
):
    key = jax.random.PRNGKey(123)
    key, init_key = jax.random.split(key)

    # 1. Initialize Model
    model = ToySuperpositionModel(num_features=num_features, hidden_dim=hidden_dim)
    dummy_input = jnp.ones((batch_size, num_features))
    params = model.init(init_key, dummy_input)["params"]

    # 2. Optimizer and State
    tx = optax.adamw(learning_rate=0.005)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Use uniform importance so the model treats all features equally
    importance = jnp.ones(num_features)

    # 3. Setup History Tracking
    save_steps = [0, 200, 1000, 2500, steps - 1]
    history = []

    # 4. Training Loop
    for step in range(steps):
        key, batch_key = jax.random.split(key)

        # Use the correlated data generator!
        x = generate_correlated_batch(
            batch_key, batch_size, num_features, sparsity, group_size
        )

        state, loss = train_step(state, x, importance)

        # Save the current state of the weights W
        if step in save_steps:
            current_W = np.asarray(state.params["W"])
            history.append((step, current_W))

        if step % 1000 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    # Now it returns BOTH the state and the history
    return state, history
