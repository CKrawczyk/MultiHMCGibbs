# Copyright 2024 Coleman Krawczyk
# SPDX-License-Identifier: Apache-2.0

'''A Numpyro Gibbs sampler that uses conditioned HMC kernels for each Gibbs step.'''

from .multihmcgibbs import MultiHMCGibbs  # noqa: F401

__version__ = '1.0.0'
