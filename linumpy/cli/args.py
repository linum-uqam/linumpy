"""Common argument-parsing helpers for linumpy CLI scripts."""

import argparse
import os
import shutil
from pathlib import Path


def get_available_cpus() -> int:
    """
    Get the number of available CPUs, respecting environment variables.

    Checks in order:
    1. LINUMPY_MAX_CPUS - maximum CPUs to use (explicit limit)
    2. LINUMPY_RESERVED_CPUS - CPUs to reserve for overhead (default: 0)

    Returns
    -------
        int: Number of available CPUs
    """
    total_cpus = os.process_cpu_count() or os.cpu_count() or 1

    # Check for explicit max CPUs limit
    max_cpus = os.environ.get("LINUMPY_MAX_CPUS")
    if max_cpus is not None:
        try:
            max_cpus = int(max_cpus)
            return max(1, min(max_cpus, total_cpus))
        except ValueError:
            pass

    # Check for reserved CPUs
    reserved = os.environ.get("LINUMPY_RESERVED_CPUS")
    if reserved is not None:
        try:
            reserved = int(reserved)
            return max(1, total_cpus - reserved)
        except ValueError:
            pass

    # Default: use all but 1 CPU
    return max(1, total_cpus - 1)


DEFAULT_N_CPUS = get_available_cpus()


def parse_processes_arg(n_processes: int | None) -> int:
    """
    Parse the n_processes argument, respecting system limits.

    Args:
        n_processes: Number of processes requested. If None or <= 0,
                     uses the default (get_available_cpus()).

    Returns
    -------
        int: Number of processes to use
    """
    available = get_available_cpus()
    if n_processes is None or n_processes <= 0 or n_processes > available:
        return available
    return n_processes


def add_processes_arg(parser: argparse.ArgumentParser | argparse._ActionsContainer) -> argparse.Action:
    """Add the ``--n_processes`` argument to *parser*."""
    a = parser.add_argument(
        "--n_processes", type=int, default=1, help="Number of processes to use. -1 to use all cores [%(default)s]."
    )
    return a


def add_overwrite_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``-f`` / ``--overwrite`` flag to *parser*."""
    parser.add_argument("-f", dest="overwrite", action="store_true", help="Force overwriting of the output files.")


def assert_output_exists(output: Path, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Error out if *output* already exists and overwrite flag is not set."""
    output_path = Path(output)
    if output_path.exists():
        if not args.overwrite:
            parser.error(f"Output {output} exists. Use -f to overwrite.")
        elif output_path.is_dir():  # remove the directory if it exists
            shutil.rmtree(output)


def add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``-v`` / ``--verbose`` argument to *parser*."""
    parser.add_argument(
        "-v",
        default="WARNING",
        const="INFO",
        nargs="?",
        choices=["DEBUG", "INFO", "WARNING"],
        dest="verbose",
        help="Produces verbose output depending on "
        "the provided level. \nDefault level is warning, "
        "default when using -v is info.",
    )


def detect_shift_units(resolution: tuple) -> tuple[float, float]:
    """Detect whether OME-Zarr resolution is in mm or µm, return (res_x_um, res_y_um).

    OME-Zarr resolution can be in mm (OME-NGFF standard) or µm depending on the writer.
    Detects by magnitude: values < 1.0 are assumed mm, >= 1.0 are assumed µm.

    Parameters
    ----------
    resolution : sequence
        Resolution tuple from read_omezarr (res_z, res_y, res_x).

    Returns
    -------
    res_x_um, res_y_um : float
        XY resolution in microns per pixel.
    """
    res_x_raw = resolution[-1]
    res_y_raw = resolution[-2] if len(resolution) >= 2 else res_x_raw

    if res_x_raw < 1.0:
        res_x_um = res_x_raw * 1000.0
        res_y_um = res_y_raw * 1000.0
    else:
        res_x_um = float(res_x_raw)
        res_y_um = float(res_y_raw)

    return res_x_um, res_y_um
