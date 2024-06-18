import copy

from collections import namedtuple, Counter
from functools import partial

from jax import device_put, jacfwd, random, value_and_grad, numpy as jnp, vmap
from numpyro.handlers import condition, seed, substitute, trace
from numpyro.infer.initialization import init_to_sample, init_to_uniform
from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import is_prng_key

MultiHMCGibbsState = namedtuple("MultiHMCGibbsState", "z, hmc_states, diverging, rng_key")
"""
 - **z** - a dict of the current latent values (all sites)
 - **hmc_states** - list of current :data:`~numpyro.infer.hmc.HMCState` (one per gibbs step)
 - **diverging** - A list of boolean value to indicate whether the current trajectory is diverging.
 - **rng_key** - random number generator seed used for the iteration.
"""


def _wrap_model(model, *args, **kwargs):
    cond_values = kwargs.pop("_cond_sites", {})
    with condition(data=cond_values), substitute(data=cond_values):
        return model(*args, **kwargs)


class MultiHMCGibbs(MCMCKernel):
    '''
    Multi-HMC-within-Gibbs.  This interface allows the user to combine multiple general purpose gradient-based
    inference (HMC or NUTS), each conditioned on a different set sub-set of sample sites, as steps in a Gibbs
    sampler.

    Note that it is the user's responsibility to ensure that every sample site is included in the
    `gibbs_sites_list` parameter and that each of the `inner_kernels` use the same Numpyro model function.

    Parameters
    ----------
    inner_kernels: List of HMC/NUTS kernels for each of the lists in `gibbs_sites`.  All kernels
        *must use the same `model`* but any of the other parameters can be different
        (e.g. `target_accept_prob`).
    gibbs_sites_list: List of lists of sites names that are *free parameters* for each Gibbs step, all other
        sample sites are fixed to their current values for the step. Each inner list is updated as a group,
        and the groups are updated in order.  All sample sites for the model must be explicitly listed in
        *only one* of the groups.

    **Example**

        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, NUTS
        >>> from MultiHMCGibbs import MultiHMCGibbs
        ...
        >>> def model():
        ...     x = numpyro.sample("x", dist.Normal(0.0, 2.0))
        ...     y = numpyro.sample("y", dist.Normal(0.0, 2.0))
        ...     numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array([1.0]))
        ...
        >>> inner_kernels = [NUTS(model), NUTS(model)]
        >>> outer_kernel = MultiHMCGibbs(inner_kernels, [['y'], ['x']])
        >>> mcmc = MCMC(kernel, num_warmup=100, num_samples=100, progress_bar=False)
        >>> mcmc.run(random.PRNGKey(0))
        >>> mcmc.print_summary()
    '''

    sample_field = "z"

    def __init__(self, inner_kernels, gibbs_sites_list):
        self.inner_kernels = []
        self.gibbs_sites_list = gibbs_sites_list
        for kdx, kernel in enumerate(inner_kernels):
            if kernel._model is not inner_kernels[0]._model:
                raise ValueError(f'inner kernel {kdx} does not have the same Numpyro model as kernel 0.')
            k = copy.copy(kernel)
            k._model = partial(_wrap_model, k.model)
            k._cond_sites = sum(
                self.gibbs_sites_list[:kdx] + self.gibbs_sites_list[kdx + 1:],
                []
            )
            self.inner_kernels.append(k)
        self._prototype_trace = None
        self._sample_fn = None

    @property
    def model(self):
        # all kernels have the same model, so just grab the first one
        return self.inner_kernels[0]._model

    @property
    def default_fields(self):
        return ("z", "diverging")

    def get_diagnostics_str(self, state):
        # show diagnostics for all inner kernels
        num_steps = '/'.join([
            '{}'.format(s.num_steps) for s in state.hmc_states
        ])
        step_size = '/'.join([
            '{:.2e}'.format(s.adapt_state.step_size) for s in state.hmc_states
        ])
        mean_accept_prob = '/'.join([
            '{:.2f}'.format(s.mean_accept_prob) for s in state.hmc_states
        ])
        return '{} steps of size {}. acc. prob={}'.format(
            num_steps, step_size, mean_accept_prob
        )

    def postprocess_fn(self, args, kwargs):
        # All kernels use the same model just conditioned on different sites.  The conditioning
        # does not change the `postprocess_fn`, so just pick one and use it in without passing in
        # a `_cond_sites` kwarg.
        _ = kwargs.pop("_cond_sites", {})
        return self.inner_kernels[0].postprocess_fn(args, kwargs)

    def check_gibbs_sites(self, model_args, model_kwargs):
        # Check that each sample site is listed exactly once in `gibbs_sites_list`
        # and provide a useful error message for any duplicate, missing, or extra sites listed.
        t = trace(
            substitute(
                seed(self.model, random.PRNGKey(0)),  # just need a trace, rng_key does not matter
                substitute_fn=init_to_uniform
            )
        ).get_trace(*model_args, **model_kwargs)
        all_sites = Counter([
            key for key, value in t.items()
            if value['type'] == 'sample' and not value['is_observed']
        ])
        listed_sites = Counter(sum(self.gibbs_sites_list, []))
        if listed_sites != all_sites:
            message = 'Expected each site to be listed **exactly once**. '
            duplicate_sites = [k for k, v in listed_sites.items() if v > 1]
            if len(duplicate_sites) > 0:
                message += f'Following sites listed more than once: {duplicate_sites}. '
            missing_sites = list((all_sites - listed_sites).keys())
            if len(missing_sites) > 0:
                message += f'Following sites in the model but not listed: {missing_sites}. '
            # remove duplicate sites before checking for extra site names
            reduced_listed_sites = Counter(listed_sites.keys())
            extra_sites = list((reduced_listed_sites - all_sites).keys())
            if len(extra_sites) > 0:
                message += f'Following sites listed but not in the model: {extra_sites}.'
            raise ValueError(message)

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        self.check_gibbs_sites(model_args, model_kwargs)

        def init_fn(init_params, key_zs):
            if (init_params is not None) and (len(init_params) == 0):
                init_params = None
            diverging = jnp.zeros(len(self.inner_kernels), dtype=bool)
            if self._prototype_trace is None:
                self._prototype_trace = trace(
                    substitute(seed(self.model, key_zs[0]), substitute_fn=init_to_sample)
                ).get_trace(*model_args, **model_kwargs)
            z = {}
            hmc_states = []
            rng_keys = []
            for key_z, kernel in zip(key_zs[1:], self.inner_kernels):
                cond_sites_kdx = {}
                init_params_kdx = {} if (init_params is not None) else None
                for name, site in self._prototype_trace.items():
                    if init_params is not None:
                        if name in kernel._cond_sites:
                            cond_sites_kdx[name] = init_params[name]
                        elif name in init_params:
                            init_params_kdx[name] = init_params[name]
                    elif name in kernel._cond_sites:
                        cond_sites_kdx[name] = site["value"]
                model_kwargs_kdx = model_kwargs | {'_cond_sites': cond_sites_kdx}
                hmc_state_kdx = kernel.init(
                    key_z,
                    num_warmup,
                    init_params=init_params_kdx,
                    model_args=model_args,
                    model_kwargs=model_kwargs_kdx
                )
                hmc_states.append(hmc_state_kdx)
                rng_keys.append(hmc_state_kdx.rng_key)
                z = z | hmc_state_kdx.z
            return MultiHMCGibbsState(z, hmc_states, diverging, jnp.stack(rng_keys))

        # not-vectorized
        if is_prng_key(rng_key):
            key_zs = random.split(rng_key, len(self.inner_kernels) + 1)
            init_state = init_fn(init_params, key_zs)
            self._sample_fn = self._sample_one
        # vectorized
        else:
            init_params = {} if init_params is None else init_params
            key_zs = vmap(
                partial(random.split, num=len(self.inner_kernels) + 1)
            )(rng_key)
            init_state = vmap(init_fn)(init_params, key_zs)
            self._sample_fn = vmap(self._sample_one, in_axes=(0, None, None))
        return device_put(init_state)

    def _sample_one(self, state, model_args, model_kwargs):
        # run step each kernel in order keeping all the sites from other
        # kernels fixed.  By the time the last kernel is stepped each
        # site will have been updated once.
        model_kwargs = {} if model_kwargs is None else model_kwargs
        postprocess_fn = self.postprocess_fn(model_args, model_kwargs)
        z = state.z
        hmc_states = []
        diverging = []
        rng_keys = []
        for hmc_state, kernel in zip(state.hmc_states, self.inner_kernels):
            # convert z to constrained space for conditioning
            z_constrained = postprocess_fn(z)
            z_cond_constrained = {
                k: v for k, v in z_constrained.items() if k in kernel._cond_sites
            }

            def potential_fn(z_hmc):
                return kernel._potential_fn_gen(
                    *model_args, _cond_sites=z_cond_constrained, **model_kwargs
                )(z_hmc)

            if kernel._forward_mode_differentiation:
                pe = potential_fn(hmc_state.z)
                z_grad = jacfwd(potential_fn)(hmc_state.z)
            else:
                pe, z_grad = value_and_grad(potential_fn)(hmc_state.z)
            hmc_state = hmc_state._replace(
                z_grad=z_grad,
                potential_energy=pe
            )
            hmc_state = kernel.sample(
                hmc_state,
                model_args,
                model_kwargs | {'_cond_sites': z_cond_constrained}
            )
            hmc_states.append(hmc_state)
            diverging.append(hmc_state.diverging)
            rng_keys.append(hmc_state.rng_key)
            # update new z values (unconstrained space)
            z = z | hmc_state.z
        return MultiHMCGibbsState(z, hmc_states, jnp.stack(diverging), jnp.stack(rng_keys))

    def sample(self, state, model_args, model_kwargs):
        return self._sample_fn(state, model_args, model_kwargs)
