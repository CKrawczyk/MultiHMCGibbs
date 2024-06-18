import unittest

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from MultiHMCGibbs import MultiHMCGibbs
from numpyro.infer import MCMC, NUTS
from numpy.testing import assert_allclose


def model():
    x = numpyro.sample("x", dist.Normal(0.0, 2.0))
    y = numpyro.sample("y", dist.Normal(0.0, 2.0))
    numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))


rng_key = jax.random.PRNGKey(0)


class TestMultiHMCGibbs(unittest.TestCase):
    def test_default(self):
        inner_kernels = [
            NUTS(model),
            NUTS(model)
        ]
        outer_kernel = MultiHMCGibbs(
            inner_kernels,
            [['y'], ['x']]
        )
        mcmc = MCMC(
            outer_kernel,
            num_warmup=1000,
            num_samples=1000,
            progress_bar=False
        )
        mcmc.run(rng_key)
        x = mcmc.get_samples()['x']
        y = mcmc.get_samples()['y']
        assert_allclose(np.mean(x), 0.5, atol=0.25, err_msg='mean(x) not close to 0.5')
        assert_allclose(np.std(x), np.sqrt(2), atol=0.25, err_msg='std(x) not close to sqrt(2)')
        assert_allclose(np.mean(y), 0.5, atol=0.25, err_msg='mean(y) not close to 0.5')
        assert_allclose(np.std(y), np.sqrt(2), atol=0.25, err_msg='std(y) not close to sqrt(2)')

    def test_sequential(self):
        inner_kernels = [
            NUTS(model),
            NUTS(model)
        ]
        outer_kernel = MultiHMCGibbs(
            inner_kernels,
            [['y'], ['x']]
        )
        mcmc = MCMC(
            outer_kernel,
            num_warmup=1000,
            num_samples=1000,
            num_chains=2,
            progress_bar=False,
            chain_method='sequential'
        )
        mcmc.run(rng_key)
        x = mcmc.get_samples()['x']
        y = mcmc.get_samples()['y']
        assert_allclose(np.mean(x), 0.5, atol=0.25, err_msg='mean(x) not close to 0.5')
        assert_allclose(np.std(x), np.sqrt(2), atol=0.25, err_msg='std(x) not close to sqrt(2)')
        assert_allclose(np.mean(y), 0.5, atol=0.25, err_msg='mean(y) not close to 0.5')
        assert_allclose(np.std(y), np.sqrt(2), atol=0.25, err_msg='std(y) not close to sqrt(2)')

    def test_vectorized(self):
        inner_kernels = [
            NUTS(model),
            NUTS(model)
        ]
        outer_kernel = MultiHMCGibbs(
            inner_kernels,
            [['y'], ['x']]
        )
        mcmc = MCMC(
            outer_kernel,
            num_warmup=1000,
            num_samples=1000,
            num_chains=2,
            progress_bar=False,
            chain_method='vectorized'
        )
        mcmc.run(rng_key)
        x = mcmc.get_samples()['x']
        y = mcmc.get_samples()['y']
        assert_allclose(np.mean(x), 0.5, atol=0.25, err_msg='mean(x) not close to 0.5')
        assert_allclose(np.std(x), np.sqrt(2), atol=0.25, err_msg='std(x) not close to sqrt(2)')
        assert_allclose(np.mean(y), 0.5, atol=0.25, err_msg='mean(y) not close to 0.5')
        assert_allclose(np.std(y), np.sqrt(2), atol=0.25, err_msg='std(y) not close to sqrt(2)')

    def test_init_params(self):
        inner_kernels = [
            NUTS(model),
            NUTS(model)
        ]
        outer_kernel = MultiHMCGibbs(
            inner_kernels,
            [['y'], ['x']]
        )
        mcmc = MCMC(
            outer_kernel,
            num_warmup=1000,
            num_samples=1000,
            progress_bar=False,
        )
        mcmc.run(rng_key, init_params={'x': jnp.array(0.0), 'y': jnp.array(0.0)})
        x = mcmc.get_samples()['x']
        y = mcmc.get_samples()['y']
        assert_allclose(np.mean(x), 0.5, atol=0.25, err_msg='mean(x) not close to 0.5')
        assert_allclose(np.std(x), np.sqrt(2), atol=0.25, err_msg='std(x) not close to sqrt(2)')
        assert_allclose(np.mean(y), 0.5, atol=0.25, err_msg='mean(y) not close to 0.5')
        assert_allclose(np.std(y), np.sqrt(2), atol=0.25, err_msg='std(y) not close to sqrt(2)')

    def test_model_mismatch(self):
        def model2():
            pass

        inner_kernels = [
            NUTS(model),
            NUTS(model2)
        ]
        with self.assertRaises(ValueError):
            MultiHMCGibbs(
                inner_kernels,
                [['y'], ['x']]
            )

    def test_missing_params(self):
        inner_kernels = [
            NUTS(model),
            NUTS(model)
        ]
        outer_kernel = MultiHMCGibbs(
            inner_kernels,
            [['y'], []]
        )
        with self.assertRaises(ValueError):
            mcmc = MCMC(
                outer_kernel,
                num_warmup=1,
                num_samples=1,
                progress_bar=False
            )
            mcmc.run(rng_key)

    def test_repeated_params(self):
        inner_kernels = [
            NUTS(model),
            NUTS(model)
        ]
        outer_kernel = MultiHMCGibbs(
            inner_kernels,
            [['y'], ['y', 'x']]
        )
        with self.assertRaises(ValueError):
            mcmc = MCMC(
                outer_kernel,
                num_warmup=1,
                num_samples=1,
                progress_bar=False
            )
            mcmc.run(rng_key)

    def test_extra_params(self):
        inner_kernels = [
            NUTS(model),
            NUTS(model)
        ]
        outer_kernel = MultiHMCGibbs(
            inner_kernels,
            [['y'], ['x', 'q']]
        )
        with self.assertRaises(ValueError):
            mcmc = MCMC(
                outer_kernel,
                num_warmup=1,
                num_samples=1,
                progress_bar=False
            )
            mcmc.run(rng_key)
