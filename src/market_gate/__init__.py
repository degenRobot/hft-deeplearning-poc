"""A read-only, deterministic market-gate teaching lab."""

__all__ = ["create_app"]


def __getattr__(name):
    # Training-only containers need NumPy/Torch, not the web runtime and its config.
    if name == "create_app":
        from .api import create_app

        return create_app
    raise AttributeError(name)
