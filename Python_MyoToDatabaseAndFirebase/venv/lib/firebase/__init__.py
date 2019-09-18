import atexit

from .async_mod import process_pool
from firebase import *


@atexit.register
def close_process_pool():
    """
    Clean up function that closes and terminates the process pool
    defined in the ``async`` file.
    """
    process_pool.close()
    process_pool.join()
    process_pool.terminate()
