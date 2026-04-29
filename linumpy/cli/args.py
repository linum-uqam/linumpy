"""General I/O helper utilities."""

import argparse
import multiprocessing
import shutil
from pathlib import Path

DEFAULT_N_CPUS = multiprocessing.cpu_count() - 1


def parse_processes_arg(n_processes: int | None) -> int:
    """Parse and clamp the number of processes to a valid range."""
    if n_processes is None or n_processes <= 0 or n_processes > DEFAULT_N_CPUS:
        return DEFAULT_N_CPUS
    return n_processes


def add_processes_arg(parser: argparse.ArgumentParser | argparse._ArgumentGroup) -> argparse.Action:
    """Add an --n_processes argument to the argument parser."""
    a = parser.add_argument(
        "--n_processes", type=int, default=1, help="Number of processes to use. -1 to use all cores [%(default)s]."
    )
    return a


def add_overwrite_arg(parser: argparse.ArgumentParser) -> None:
    """Add a -f overwrite flag to the argument parser."""
    parser.add_argument("-f", dest="overwrite", action="store_true", help="Force overwriting of the output files.")


def assert_output_exists(output: Path, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Raise a parser error if the output already exists and overwrite is not set."""
    if Path(output).exists():
        if not args.overwrite:
            parser.error(f"Output {output} exists. Use -f to overwrite.")
        elif Path(output).is_dir():  # remove the directory if it exists
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
