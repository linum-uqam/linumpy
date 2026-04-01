"""linumpy — light-sheet and OCT microscopy processing library."""

# Configure thread limits FIRST, before any numerical libraries are imported
from linumpy.config.threads import (
    apply_threadpool_limits as apply_threadpool_limits,
)
from linumpy.config.threads import (
    configure_all_libraries as configure_all_libraries,
)
from linumpy.config.threads import (
    configure_sitk as configure_sitk,
)
from linumpy.config.threads import (
    configure_thread_limits as configure_thread_limits,
)

import os as _os
from pathlib import Path as _Path


def get_home():
    """Set a user-writeable file-system location to put files."""
    if "LINUMPY_HOME" in _os.environ:
        return _os.environ["LINUMPY_HOME"]
    return str(_Path.home() / ".linumpy")


def get_root():
    return str(_Path(__file__).resolve().parent.parent)


LINUMPY_HOME = get_home()
LINUMPY_ROOT = get_root()

# Apply runtime thread pool limits (for libraries that don't respect env vars)
apply_threadpool_limits()

# Note: configure_sitk() must be called after SimpleITK is imported.
# Scripts that use SimpleITK should call:
#   from linumpy.config.threads import configure_all_libraries
#   configure_all_libraries()  # after all imports
