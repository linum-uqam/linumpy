"""Common argument-parsing helpers for linumpy CLI scripts."""

import argparse
import multiprocessing
import os
import shutil
from pathlib import Path


def get_available_cpus() -> int:
    """Get the number of available CPUs, respecting environment variables."""
    total_cpus = multiprocessing.cpu_count()

    max_cpus = os.environ.get("LINUMPY_MAX_CPUS")
    if max_cpus is not None:
        try:
            max_cpus_int = int(max_cpus)
            return max(1, min(max_cpus_int, total_cpus))
        except ValueError:
            pass

    reserved = os.environ.get("LINUMPY_RESERVED_CPUS")
    if reserved is not None:
        try:
            reserved_int = int(reserved)
            return max(1, total_cpus - reserved_int)
        except ValueError:
            pass

    return max(1, total_cpus - 1)


DEFAULT_N_CPUS = get_available_cpus()


def parse_processes_arg(n_processes: int | None) -> int:
    """Parse the n_processes argument, respecting system limits."""
    available = get_available_cpus()
    if n_processes is None or n_processes <= 0 or n_processes > available:
        return available
    return n_processes


def add_processes_arg(parser: argparse.ArgumentParser | argparse._ArgumentGroup) -> argparse.Action:
    """Add the ``--n_processes`` argument to *parser*."""
    return parser.add_argument(
        "--n_processes", type=int, default=1, help="Number of processes to use. -1 to use all cores [%(default)s]."
    )


def add_overwrite_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``-f`` / ``--overwrite`` flag to *parser*."""
    parser.add_argument("-f", dest="overwrite", action="store_true", help="Force overwriting of the output files.")


def assert_output_exists(output: Path, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Raise a parser error if the output already exists and overwrite is not set."""
    if Path(output).exists():
        if not args.overwrite:
            parser.error(f"Output {output} exists. Use -f to overwrite.")
        elif Path(output).is_dir():
            shutil.rmtree(output)


def add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    """Add a -v verbose argument to the argument parser."""
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
