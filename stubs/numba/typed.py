"""Shim for ``numba.typed`` (only ``List`` is used by shap)."""


class List(list):
    """Drop-in replacement for ``numba.typed.List`` backed by a plain list."""

    @classmethod
    def empty_list(cls, *args, **kwargs):
        return cls()
