"""Minimal no-op ``numba`` shim.

Smart App Control is enabled on this machine and blocks ``llvmlite.dll`` (an
unsigned binary that the real numba depends on), so numba cannot be imported.
``shap`` imports numba only for optional JIT speedups of a few clustering
helpers, so this shim provides passthrough implementations. Correctness is
unaffected; only the raw speed of those helpers is reduced, which is negligible
at this project's scale.
"""

__version__ = "0.0.0+famha-shim"


def njit(*args, **kwargs):
    """No-op replacement for ``numba.njit`` supporting both call styles."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(func):
        return func

    return decorator


# ``jit`` behaves the same as ``njit`` for our purposes.
jit = njit


def prange(*args, **kwargs):
    return range(*args, **kwargs)


from . import typed  # noqa: E402,F401
