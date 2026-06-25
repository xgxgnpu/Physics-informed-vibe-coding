"""Ablation control flags, read once per process from the environment.

ABL_STRATEGY : frequency layering strategy. 'default' == each case's original
               (E11) sampler, so default + scale 1.0 reproduces E11 exactly.
ABL_SCALE    : multiplicative scale on ALL sampled (frozen) frequencies, used
               for the characteristic-frequency-magnitude (w_char) sweep.
"""
import os

STRATEGY = os.environ.get("ABL_STRATEGY", "default")
SCALE = float(os.environ.get("ABL_SCALE", "1.0"))
