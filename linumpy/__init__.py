def get_home():
    import os
    """ Set a user-writeable file-system location to put files. """
    if 'LINUMPY_HOME' in os.environ:
        return os.environ['LINUMPY_HOME']
    return os.path.join(os.path.expanduser('~'), '.linumpy')


def get_root():
    import os
    return os.path.realpath(f"{os.path.dirname(os.path.abspath(__file__))}/..")


LINUMPY_HOME = get_home()
LINUMPY_ROOT = get_root()
