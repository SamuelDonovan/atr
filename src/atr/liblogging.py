# From the python standard library
import logging


def setup_logger(verbosity):
    log_levels = {
        0: logging.CRITICAL,
        1: logging.ERROR,
        2: logging.WARN,
        3: logging.INFO,
        4: logging.DEBUG,
    }

    logging.basicConfig(
        level=log_levels[verbosity],
        format="[%(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )


if __name__ == "__main__":
    raise Exception("This module is not an entry point!")
