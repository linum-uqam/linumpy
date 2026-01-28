# -*- coding:utf8 -*-
import multiprocessing
import os
import shutil


def get_available_cpus():
    """
    Get the number of available CPUs, respecting environment variables.

    Checks in order:
    1. LINUMPY_MAX_CPUS - maximum CPUs to use (explicit limit)
    2. LINUMPY_RESERVED_CPUS - CPUs to reserve for overhead (default: 0)

    Returns:
        int: Number of available CPUs
    """
    total_cpus = multiprocessing.cpu_count()

    # Check for explicit max CPUs limit
    max_cpus = os.environ.get('LINUMPY_MAX_CPUS')
    if max_cpus is not None:
        try:
            max_cpus = int(max_cpus)
            return max(1, min(max_cpus, total_cpus))
        except ValueError:
            pass

    # Check for reserved CPUs
    reserved = os.environ.get('LINUMPY_RESERVED_CPUS')
    if reserved is not None:
        try:
            reserved = int(reserved)
            return max(1, total_cpus - reserved)
        except ValueError:
            pass

    # Default: use all but 1 CPU
    return max(1, total_cpus - 1)


DEFAULT_N_CPUS = get_available_cpus()


def parse_processes_arg(n_processes):
    """
    Parse the n_processes argument, respecting system limits.

    Args:
        n_processes: Number of processes requested. If None or <= 0,
                     uses the default (get_available_cpus()).

    Returns:
        int: Number of processes to use
    """
    available = get_available_cpus()
    if n_processes is None or n_processes <= 0:
        return available
    elif n_processes > available:
        return available
    return n_processes


def add_processes_arg(parser):
    a = parser.add_argument('--n_processes', type=int, default=1,
                            help='Number of processes to use. -1 to use '
                                 'all cores [%(default)s].')
    return a


def add_overwrite_arg(parser):
    parser.add_argument(
        '-f', dest='overwrite', action='store_true', help='Force overwriting of the output files.')


def assert_output_exists(output, parser, args):
    if os.path.exists(output):
        if not args.overwrite:
            parser.error(f'Output {output} exists. Use -f to overwrite.')
        elif os.path.isdir(output):  # remove the directory if it exists
            shutil.rmtree(output)


def add_verbose_arg(parser):
    parser.add_argument('-v', default="WARNING", const='INFO', nargs='?',
                        choices=['DEBUG', 'INFO', 'WARNING'], dest='verbose',
                        help='Produces verbose output depending on '
                             'the provided level. \nDefault level is warning, '
                             'default when using -v is info.')
