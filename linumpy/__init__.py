"""linumpy — light-sheet and OCT microscopy processing library."""

# Configure thread limits FIRST, before any numerical libraries are imported
from linumpy.config.threads import apply_threadpool_limits

import os as _os
from pathlib import Path as _Path


def get_home() -> str:
    """Set a user-writeable file-system location to put files."""
    if "LINUMPY_HOME" in _os.environ:
        return _os.environ["LINUMPY_HOME"]
    return str(_Path.home() / ".linumpy")


def get_root() -> str:
    """Return the linumpy package root directory."""
    return str((_Path(__file__).resolve().parent / "..").resolve())


LINUMPY_HOME = get_home()
LINUMPY_ROOT = get_root()

# Apply runtime thread pool limits (for libraries that don't respect env vars)
apply_threadpool_limits()

# Note: configure_sitk() must be called after SimpleITK is imported.
# Scripts that use SimpleITK should call configure_all_libraries() after all imports.
