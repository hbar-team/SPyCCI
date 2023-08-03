import subprocess, logging

from os.path import abspath
from os import environ

logger = logging.getLogger(__name__) 

def locate_executable(name: str) -> str:
    """
    Locates a given executable in the system PATH. If the program is found the path is
    returned, else an exception is raised.

    Arguments
    ---------
    name: str
        The name of the program to locate.

    Raises
    ------
    RuntimeError
        Exception raised if the program is not found in the system path.

    Returns
    -------
    str
        The path to the requested program.
    """
    try:
        output = subprocess.check_output(f"which {name}", shell=True).decode("utf-8")
        output = output.strip("\n")
        return output

    except:
        raise RuntimeError(f"cannot find '{name}' in the system path")


def locate_vmd() -> str:
    """
    Locate the path to the 'vmd' folder from the system PATH.

    Returns
    -------
    str
        The path to the vmd base folder (the one containing the `bin` and `lib` subfolders)
    """
    path = locate_executable("vmd")
    return path.rstrip("/bin/vmd")


def locate_orca(version: str = None, get_folder: bool = False) -> str:
    """
    Locates the path to the 'orca' executable from the system PATH. If the executable is
    located checks that the correct version of OpenMPI is exported (explicit reference to
    the static builds). If specified, checks that the correct version of orca is loaded.

    Arguments
    ---------
    version: str
        The string defining the desired version of orca. If set to None (default) all
        versions of orca are accepted.
    get_folder: bool
        If set to True will return the path of the folder containing the orca executable
        instead of the default path to the executable itself. Equivalent to applying an
        `rstrip('/orca')` to the executable path

    Returns
    -------
    str
        The path to the orca executable file.
    """
    path = locate_executable("orca")

    # Check if the available version of orca matches the requirements
    orca_version = None
    output = subprocess.run(["orca", "--version"], capture_output=True, text=True).stdout
    for line in output.split("\n"):

        if "Program Version" in line:
            orca_version: str = line.split()[2]
            break

    if orca_version is None:
        raise RuntimeError("Failed to read the version of the orca software.")

    elif version is not None and orca_version != version:
        raise RuntimeError(
            f"The required orca version is not available. Version {orca_version} found instead."
        )

    # Check if a MPI version is available in the system PATH
    _ = locate_executable("mpirun")

    # Check if the version of the loaded OpenMPI
    openmpi_version = None
    output = subprocess.run(["mpirun", "--version"], capture_output=True, text=True).stdout

    for line in output.split("\n"):

        if "(Open MPI)" in line:
            openmpi_version: str = line.split()[3]
            break

    if openmpi_version is None:
        raise RuntimeError("OpenMPI is either not available or the version cannot be found")

    # Check if the available version meets the requirements
    openmpi_required = {"5.0.*": ["4.1.1"], "4.2.*": ["3.1.4"], "4.1.*": ["3.1.3", "2.1.5"]}

    key = ".".join(orca_version.split(".")[0:-1] + ["*"])
    if openmpi_version not in openmpi_required[key]:
        msg = " or ".join(openmpi_required[key])
        raise RuntimeError(
            f"orca {orca_version} retuires OpenMPI {msg}. OpenMPI {openmpi_version} found instead."
        )
    
    try:
        folder = path.rstrip("/orca")
        locate_executable(f"{folder}/otool_xtb")
    except:
        logger.warning("The `otool_xtb` symbolic link to xTB has not been found. The orca interface for xTB will not be available.")

    return path.rstrip("/orca") if get_folder is True else path


def locate_xtb(version: str = None) -> str:
    """
    Locates the path to the 'xtb' executable from the system PATH. If specified, checks that
    the correct version of xtb is loaded.

    Arguments
    ---------
    version: str
        The string defining the desired version of xtb. If set to None (default) all
        versions of xtb are accepted.

    Returns
    -------
    str
        The path to the xtb executable file.
    """
    path = locate_executable("xtb")

    # Check if the available version of xtb matches the requirements
    xtb_version = None
    output = subprocess.run(["xtb", "--version"], capture_output=True, text=True).stdout
    for line in output.split("\n"):

        if "xtb version" in line:
            xtb_version: str = line.split()[3]
            break

    if xtb_version is None:
        raise RuntimeError("Failed to read the version of the xtb software.")

    elif version is not None and xtb_version != version:
        raise RuntimeError(
            f"The required xtb version is not available. Version {xtb_version} found instead."
        )

    return path


def locate_crest(version: str = None) -> str:
    """
    Locates the path to the 'crest' executable from the system PATH. If specified, checks
    that the correct version of crest is loaded.

    Arguments
    ---------
    version: str
        The string defining the desired version of crest. If set to None (default) all
        versions of crest are accepted.

    Returns
    -------
    str
        The path to the crest executable file.
    """
    path = locate_executable("crest")

    # Check if the available version of crest matches the requirements
    crest_version = None
    output = subprocess.run(["crest", "--version"], capture_output=True, text=True).stdout
    for line in output.split("\n"):

        if "Version" in line:
            crest_version: str = line.split()[1]
            break

    if crest_version is None:
        raise RuntimeError("Failed to read the version of the crest software.")

    elif version is not None and crest_version != version:
        raise RuntimeError(
            f"The required crest version is not available. Version {crest_version} found instead."
        )

    return path


def locate_dftbplus(version: str = None) -> str:
    """
    Locates the path to the 'dftb+' executable from the system PATH. If specified, checks
    that the correct version of dftb+ is loaded.

    Arguments
    ---------
    version: str
        The string defining the desired version of dftb+. If set to None (default) all
        versions of dftb+ are accepted.

    Returns
    -------
    str
        The path to the dftb+ executable file.
    """
    path = locate_executable("dftb+")

    # Check if the available version of dftb+ matches the requirements
    dftbplus_version = None
    dftbplus_output = subprocess.run(
        ["dftb+", "--version"], capture_output=True, text=True
    ).stdout
    for line in dftbplus_output.split("\n"):

        if "DFTB+ release" in line:
            dftbplus_version: str = line.split()[3]
            break

    if dftbplus_version is None:
        raise RuntimeError("Failed to read the version of the dftb+ software.")

    elif version is not None and dftbplus_version != version:
        raise RuntimeError(
            f"The required dftb+ version is not available. Version {dftbplus_version} found instead."
        )

    return path


def locate_dftbparamdir() -> str:
    """
    Locates the path to the DFTBPLUS_PARAM_DIR environment variable.

    Returns
    -------
    str
        The path to the DFTBPLUS_PARAM_DIR environment variable.
    """
    path = None
    try:
        path = environ["DFTBPLUS_PARAM_DIR"]
    except KeyError:
        raise RuntimeError("Failed to locate DFTBPLUS_PARAM_DIR environment variable.")

    return path
