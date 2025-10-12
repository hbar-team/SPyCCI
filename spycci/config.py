import os
import logging
from enum import IntEnum

logger = logging.getLogger(__name__)

__JSON_VERSION__ = 1

MPI_FLAGS = "--bind-to none"

global VERSION_MATCH
# Used in dependency_finder.py to match version maps
# Options: "strict", "minor", "major", "disabled"
# "strict": match exact version (e.g., 4.1.6 -> 4.1.6)
# "minor": allow minor version changes (e.g., 4.1.8 -> 4.1.x, 4.1 -> 4.x)
# "major": match only major version (e.g., 4.1.8 -> 4.x)
# "disabled": do not check versions at all (print a warning)
VERSION_MATCH = os.environ.get("SPYCCI_VERSION_MATCH", "strict").lower()
if VERSION_MATCH not in ["strict", "minor", "major", "disabled"]:
    raise ValueError(
        f"Invalid value for SPYCCI_VERSION_MATCH: {VERSION_MATCH}. "
        'Allowed values are "strict", "minor", "major", "disabled".'
    )


def get_ncores():
    try:
        ncores = int(os.environ["OMP_NUM_THREADS"])
        logger.debug("Environment variable OMP_NUM_THREADS found")
    except:
        ncores = len(os.sched_getaffinity(0))
        logger.debug("Environment variable OMP_NUM_THREADS not found")

    logger.debug(f"Number of cores: {ncores}")
    return ncores


class StrictnessLevel(IntEnum):
    NORMAL = 0
    STRICT = 1
    VERY_STRICT = 2


STRICTNESS_LEVEL = StrictnessLevel.NORMAL
