"""Legitimacy Gating in Social Learning -- Paper II reference implementation.

A computational model in which reward feedback disciplines beliefs only to the
extent that the reward rule is treated as legitimate. Self-contained; shares no
code with Paper I.
"""
from . import config, network, confidence, fairness, modal, model, metrics, scenarios, meanfield
from .config import Params, SCALE

__all__ = ["config", "network", "confidence", "fairness", "modal", "model",
           "metrics", "scenarios", "Params", "SCALE"]
__version__ = "1.0.0"
