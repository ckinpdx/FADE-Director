"""
patch.py
Apply runtime patches to a ComfyUI workflow dict.

ComfyUI workflows are nested dicts: workflow[node_id]["inputs"][field] = value.
This module deep-copies the template and applies a flat patches dict so the
original template is never mutated.

Usage:
    from backend.comfyui.patch import apply

    patched = apply(template, {
        "149": {"text": "a woman in a red leather jacket"},
        "160": {"seed": 42},
        "166": {"width": 1152, "height": 2048},
        "177": {"filename_prefix": "sessions/abc/images/scene_1"},
    })
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


def apply(
    workflow: dict,
    patches:  dict[str, dict[str, Any]],
) -> dict:
    """
    Deep-copy a ComfyUI workflow dict and apply field-level patches.

    Args:
        workflow: Loaded workflow JSON as a Python dict (not mutated).
        patches:  {node_id: {input_field: value}} — every entry sets
                  workflow[node_id]["inputs"][field] = value.

    Returns:
        A new dict with patches applied. The original is unchanged.

    Raises:
        KeyError: If a node_id in patches does not exist in the workflow.
    """
    patched = copy.deepcopy(workflow)

    for node_id, fields in patches.items():
        if node_id not in patched:
            raise KeyError(
                f"Node '{node_id}' not found in workflow. "
                f"Available nodes: {list(patched.keys())[:10]} ..."
            )
        for field, value in fields.items():
            patched[node_id]["inputs"][field] = value
            logger.debug("patch node=%s  field=%s  value=%r", node_id, field, _abbrev(value))

    return patched


def _abbrev(value: Any, maxlen: int = 60) -> str:
    """Abbreviate long values for debug logging."""
    s = repr(value)
    return s if len(s) <= maxlen else s[:maxlen] + "..."
