# -*- coding:utf8 -*-
import multiprocessing


DEFAULT_N_CPUS = multiprocessing.cpu_count() - 1


def parse_processes_arg(n_processes):
    if n_processes is None or n_processes <= 0:
        return DEFAULT_N_CPUS
    elif n_processes > DEFAULT_N_CPUS:
        return DEFAULT_N_CPUS
    return n_processes


def add_processes_arg(parser):
    a = parser.add_argument('--n_processes', type=int,
                            help=f'Number of processes to use [{DEFAULT_N_CPUS}].')
    return a
