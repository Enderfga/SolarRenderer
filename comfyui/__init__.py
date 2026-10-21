"""SolarRenderer for ComfyUI (see comfyui/README.md)."""

from .causal import SolarRendererCausalSampler
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS["SolarRendererCausalSampler"] = SolarRendererCausalSampler
NODE_DISPLAY_NAME_MAPPINGS["SolarRendererCausalSampler"] = "SolarRenderer Causal Sampler (long clips)"

try:  # the page needs the running server; the nodes load without it (tests, tooling)
    from . import app  # noqa: F401
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
