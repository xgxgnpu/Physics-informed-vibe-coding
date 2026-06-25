"""Uniform frequency-layering strategies for the E16 ablation.

Each case's ORIGINAL (E11) sampler is the 'default' three-level strategy and is
left untouched in the case file; these alternatives are only used when
ABL_STRATEGY != 'default'. They are parameterised by the case's characteristic
frequency `wc` and an upper frequency `fmax`, so the comparison is fair and
consistent across the heterogeneous cases.
"""
import jax
import jax.numpy as jnp


def strategy_sample(key, K, wc, fmax, strategy):
    wc = float(wc)
    fmax = float(fmax)
    k1, k2, k3 = jax.random.split(key, 3)
    if fmax > wc * 1.05:
        hi_lo, hi_hi = wc, fmax
    else:
        hi_lo, hi_hi = 0.5 * wc, wc

    if strategy == "S1_single":
        return jnp.full((K,), wc)

    if strategy == "S2_two":
        nl = K // 2
        low = jnp.linspace(1.0, wc, nl)
        char = jnp.abs(jax.random.normal(k2, (K - nl,)) * 0.1 * wc + wc)
        return jnp.sort(jnp.concatenate([low, char]))

    if strategy == "S4_continuous":
        return jnp.sort(jax.random.uniform(k1, (K,), minval=1.0, maxval=hi_hi))

    if strategy == "S5_40_40_20":
        nl = int(round(0.4 * K))
        nc = int(round(0.4 * K))
        nh = K - nl - nc
        low = jnp.linspace(1.0, wc, nl)
        char = jnp.abs(jax.random.normal(k2, (nc,)) * 0.1 * wc + wc)
        high = jax.random.uniform(k3, (nh,), minval=hi_lo, maxval=hi_hi)
        return jnp.sort(jnp.concatenate([low, char, high]))

    raise ValueError("unknown strategy: %s" % strategy)
