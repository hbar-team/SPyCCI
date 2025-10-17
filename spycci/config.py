import os
import logging
from enum import IntEnum

logger = logging.getLogger(__name__)

def get_ncores():
    try:
        ncores = int(os.environ["OMP_NUM_THREADS"])
        logger.debug("Environment variable OMP_NUM_THREADS found")
    except:
        ncores = len(os.sched_getaffinity(0))
        logger.debug("Environment variable OMP_NUM_THREADS not found")

    logger.debug(f"Number of cores: {ncores}")
    return ncores

# The version of the internal SPyCCI JSON file format
__JSON_VERSION__ = 1

# The string encoding the default options to be used during an MPI call
MPI_FLAGS = "--bind-to none"

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

# Strictness of the Level of theory check performed by the properties.py module
# Options: "NORMAL", "STRICT"
# "NORMAL" : The electronic and vibrational levels of theory can be different
# "STRICT" : The electronic and vibrational levels of theory cannot differ within a property object
class StrictnessLevel(IntEnum):
    NORMAL = 0
    STRICT = 1

STRICTNESS_LEVEL = StrictnessLevel.NORMAL
