import copy

from collections import namedtuple
from functools import partial

from jax import device_put, jacfwd, random, value_and_grad, numpy as jnp, vmap
from numpyro.handlers import condition, seed, substitute, trace
from numpyro.infer.initialization import init_to_sample
from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import is_prng_key

MultiHMCGibbsState = namedtuple("MultiHMCGibbsState", "z, hmc_states, diverging, rng_key")
"""
 - **z** - a dict of the current latent values (all sites)
 - **hmc_states** - list of current :data:`~numpyro.infer.hmc.HMCState`
 - **diverging** - A list of boolean value to indicate whether the current trajectory is diverging.
"""


def _wrap_model(model, *args, **kwargs):
    cond_values = kwargs.pop("_cond_sites", {})
    with condition(data=cond_values), substitute(data=cond_values):
        return model(*args, **kwargs)


class MultiHMCGibbs(MCMCKernel):
    sample_field = "z"

    def __init__(self, inner_kernels, gibbs_sites_list):
        '''
        Parameters
        ----------
        inner_kernels: List of HMC/NUTS kernels for each of the lists in `gibbs_sites`.  All kernels
            must use the same `model` but can any of the other parameters be different (e.g.
            `target_accept_prob`).
        gibbs_sites_list: List of lists of sites names to be gibbs stepped over.  Each inner list
            is updated as a group, and the groups are updated in order.  All sites for the model
            must be explicitly listed in one of the groups.
        '''
        self.inner_kernels = []
        self.gibbs_sites_list = gibbs_sites_list
        for kdx, kernel in enumerate(inner_kernels):
            k = copy.copy(kernel)
            k._model = partial(_wrap_model, k.model)
            k._cond_sites = sum(
                self.gibbs_sites_list[:kdx] + self.gibbs_sites_list[kdx+1:],
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

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()

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
