import jax.numpy as jnp
import flax.linen as nn
from typing import Literal
from core.activations import solu


# 1. Define the Model using Flax
class ToySuperpositionModel(nn.Module):
    num_features: int
    hidden_dim: int
    activation_name: Literal["relu", "solu"] = "relu"

    def setup(self):
        # W projects from the larger feature space down to the hidden bottleneck
        self.W = self.param(
            "W", nn.initializers.xavier_normal(), (self.num_features, self.hidden_dim)
        )
        # Bias applied before the ReLU
        self.b = self.param("b", nn.initializers.zeros, (self.num_features,))
        
        # define the activation function
        if self.activation_name == "solu":
            self.activation_fn = solu
        else:
            self.activation_fn = nn.relu

    def __call__(self, x):
        # 1. Project down to hidden dimension: h = xW -> shape (batch_size, hidden_dim)
        h = jnp.dot(x, self.W)

        # 2. Project back up and apply ReLU: x' = activation(hW^T + b)
        x_prime = self.activation_fn(jnp.dot(h, self.W.T) + self.b)

        return x_prime
