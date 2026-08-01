"""Global gradient-checkpointing switch for the model's memory-heavy blocks.

Activation checkpointing (image encoder, feature encoder, query decoder) trades compute
for memory and is on by default. On large-memory hardware it can be disabled to skip the
recompute in backward. The flag is process-global because checkpointing is a training-mode
decision, not a per-instance one.
"""

_ENABLED = True


def set_grad_checkpointing(enabled: bool) -> None:
    """Enable or disable activation checkpointing for the whole process."""
    global _ENABLED
    _ENABLED = enabled


def grad_checkpointing_enabled() -> bool:
    """Whether activation checkpointing is currently on."""
    return _ENABLED
