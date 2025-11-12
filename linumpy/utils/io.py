# -*- coding:utf8 -*-
import multiprocessing
import os
import shutil

DEFAULT_N_CPUS = multiprocessing.cpu_count() - 1


def parse_processes_arg(n_processes):
    if n_processes is None or n_processes <= 0:
        return DEFAULT_N_CPUS
    elif n_processes > DEFAULT_N_CPUS:
        return DEFAULT_N_CPUS
    return n_processes


def add_processes_arg(parser):
    a = parser.add_argument('--n_processes', type=int, default=1,
                            help='Number of processes to use. -1 to use '
                                 'all cores [%(default)s].')
    return a


def add_overwrite_arg(parser):
    parser.add_argument('-f', dest='overwrite', action='store_true',
                        help='Force overwriting of the output files.')


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
