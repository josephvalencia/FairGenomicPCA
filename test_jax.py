import jax.numpy as jnp
from jax import grad, jit, vmap


def square(x):

    return 0.5 * jnp.power(x,3)


x_small = jnp.array(3.)
derivative_fn = grad(square)
print(derivative_fn(x_small))
