import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def visualize_interference_jax(state):
    # 1. Extract the weight matrix from the Flax TrainState
    # The parameters are stored in the state.params dictionary
    W = state.params["W"]

    # 2. Calculate the interference matrix W @ W^T
    # Compute this using JAX numpy, then convert to standard NumPy for plotting
    interference_matrix = jnp.dot(W, W.T)

    # JAX arrays can usually be passed directly to matplotlib,
    # but explicitly converting to numpy is best practice to avoid tracing issues
    interference_matrix_np = np.asarray(interference_matrix)

    # 3. Set up the plot
    plt.figure(figsize=(8, 6))

    # 4. Plot the heatmap using a diverging colormap
    # 'bwr' (blue-white-red) ensures 0 (no interference) maps perfectly to white
    im = plt.imshow(interference_matrix_np, cmap="bwr", vmin=-1.0, vmax=1.0)

    # 5. Formatting
    plt.colorbar(im, label="Dot Product (Interference)")
    plt.title("Feature Interference Matrix ($W W^T$) - JAX/Flax")
    plt.xlabel("Features (sorted by importance)")
    plt.ylabel("Features (sorted by importance)")

    # Add minor gridlines to delineate individual features cleanly
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, interference_matrix_np.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, interference_matrix_np.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    plt.show()
