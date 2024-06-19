[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12167630.svg)](https://doi.org/10.5281/zenodo.12167630)


# An HMC-within-Gibbs sampler for Numpyro

This package adds a new HMC-within-Gibbs sampler to Numpyro.  Unlike the `HMCGibbs` sampler currently available, this sampler is for situations where you do not have an analytic form for one of your conditioned distributions.  Instead, it uses an HMC/NUTS sampler to estimate draws from *each* of the conditioned distributions.

To use `MultiHMCGibbs` you need to create a list of HMC or NUTS kernels that wrap the same model, but each can have its own keywords such as `target_accept_prob` or `max_tree_depth`.  The other argument is a list of lists containing the **free** parameters for each of the inner kernels. 

Internally the sampler will:
1. Loop over the kernels in the list
2. Conditioned it on the non-free parameters
3. Re-calculate the likelihood and gradients at the new conditioned point
4. Step the kernel forward
5. Move on to the next kernel

Documentation: [https://ckrawczyk.github.io/MultiHMCGibbs/](https://ckrawczyk.github.io/MultiHMCGibbs/)

GitHub: [https://github.com/CKrawczyk/MultiHMCGibbs](https://github.com/CKrawczyk/MultiHMCGibbs)

## Installation

You can install the package with `pip` after cloning the repository.

```bash
git clone https://github.com/CKrawczyk/MultiHMCGibbs.git
cd MultiHMCGibbs
pip install .
```

## Example usage

```python
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from MultiHMCGibbs import MultiHMCGibbs

def model():
     x = numpyro.sample("x", dist.Normal(0.0, 2.0))
     y = numpyro.sample("y", dist.Normal(0.0, 2.0))
     numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))

inner_kernels = [
    NUTS(model),
    NUTS(model)
]
outer_kernel = MultiHMCGibbs(
    inner_kernels,
    [['y'], ['x']]
)
mcmc = MCMC(
    kernel,
    num_warmup=100,
    num_samples=100,
    progress_bar=False
)
mcmc.run(random.PRNGKey(0))
mcmc.print_summary()
```

## Contributing

Install all the development dependencies:
```bash
pip install -e .[dev]
```

Run tests with:
```bash
coverage run
coverage report
```

Build documentation with:
```bash
./build_docs
```
