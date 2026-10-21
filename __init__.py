"""ComfyUI loads this repository as a custom node pack when it is cloned into ComfyUI/custom_nodes (see comfyui/)."""

from .comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
