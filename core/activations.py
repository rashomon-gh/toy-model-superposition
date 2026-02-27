import jax.numpy as jnp
import flax.linen as nn


def solu(x, axis=-1) -> jnp.ndarray:
    out = x * nn.softmax(x, axis=axis)
    out = nn.LayerNorm()(out)
    return out
    
    
