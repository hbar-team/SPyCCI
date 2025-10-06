import os
import logging

logger = logging.getLogger(__name__)

__JSON_VERSION__ = 1

global MPI_FLAGS
MPI_FLAGS = "--bind-to none"

global VERSION_MATCH
# Used in dependency_finder.py to match version maps
# Options: "default", "strict", "loose"
# "default": allow minor version changes (e.g., 4.1.8 -> 4.1.x, 4.1 -> 4.x)
# "strict": match exact version (e.g., 4.1.6 -> 4.1.6)
# "loose": match only major version (e.g., 4.1.8 -> 4.x)
# "disabled": do not check versions at all (print a warning)
VERSION_MATCH = os.environ.get("SPYCCI_VERSION_MATCH", "default")
if VERSION_MATCH not in ["default", "strict", "loose", "disabled"]:
    raise ValueError(
        f"Invalid value for SPYCCI_VERSION_MATCH: {VERSION_MATCH}. "
        'Allowed values are "default", "strict", "loose", "disabled".'
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
