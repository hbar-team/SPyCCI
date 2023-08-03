import os
import shutil
from rdkit import Chem
from copy import deepcopy
from spycci.systems import System
import logging

logger = logging.getLogger(__name__)


def cyclization_check(start_file: str, end_file: str):
    """Checks if a cyclization has occurred (e.g., during a
    geometry optimization), or if a ring opening has occurred.

    Parameters
    ----------
    start_file : str
        .xyz file containing the starting geometry
    end_file : str
        .xyz file containing the final geometry

    Returns
    -------
    If a change in the number of rings is detected, returns True.
    """

    os.system(f"crest --testtopo {start_file} > start_topo.out 2>> start_topo.out")
    os.system(f"crest --testtopo {end_file} > end_topo.out 2>> end_topo.out")

    with open("start_topo.out", "r") as out:
        start_rings = 0
        for line in out:
            if "Total number of rings in the system" in line:
                start_rings = int(line.split()[-1])
                break

    with open("end_topo.out", "r") as out:
        end_rings = 0
        for line in out:
            if "Total number of rings in the system" in line:
                end_rings = int(line.split()[-1])
                break
            if "No (valid) input file!" in line:
                break

    if start_rings != end_rings:
        return True
    else:
        return False


def dissociation_check():
    """Checks if a dissociation has occurred (e.g., during a geometry optimization).
    Requires the presence of a .mol file in the current directory.

    Returns
    -------
    True, if a dissociation is detected
    False, if a dissociation is not detected
    """

    mol_file = [f for f in os.listdir(".") if f.endswith(".mol")][-1]
    end_mol = Chem.MolFromMolFile(
        mol_file, sanitize=False, removeHs=False, strictParsing=False
    )
    end_smiles = Chem.MolToSmiles(end_mol)

    if "." in end_smiles:
        return True
    else:
        return False


def process_output(
    mol: System,
    method: str,
    calc: str,
    charge: int = None,
    spin: int = None,
    save_cubes: bool = False,
):
    """Processes the output of a calculation, copying the output files to a safe directory
    (output_files) in the parent directory tree, and cleans the temporary directory if requested.

    Parameters
    ----------
    mol : System object
        Systems processed in the calculation
    method : str
        String indicating the level of theory for the calculation
    calc : str
        Type of calculation
    charge : int, optional
        Charge of the molecule in the calculation
    spin : int, optional
        Spin of the molecule in the calculation
    save_cubes: bool
        If set to true will copy the `.cube` files in a safe directory (`output_densities`)
        in the parent directory tree
    """

    if charge is None:
        charge = mol.charge
    if spin is None:
        spin = mol.spin

    os.makedirs("../output_files", exist_ok=True)

    if os.path.exists("output.out"):
        shutil.copy(
            "output.out",
            f"../output_files/{mol.name}_{charge}_{spin}_{method}_{calc}.out",
        )

    if os.path.exists("output.err"):
        os.makedirs("../error_files", exist_ok=True)
        shutil.copy(
            "output.err",
            f"../error_files/{mol.name}_{charge}_{spin}_{method}_{calc}.err",
        )

    if save_cubes:

        os.makedirs("../output_densities", exist_ok=True)

        if os.path.exists("eldens.cube"):
            shutil.copy(
                "eldens.cube",
                f"../output_densities/{mol.name}_{charge}_{spin}_{method}_{calc}.eldens.cube",
            )

        if os.path.exists("density.cub"):
            shutil.copy(
                "density.cub",
                f"../output_densities/{mol.name}_{charge}_{spin}_{method}_{calc}.eldens.cube",
            )

        if os.path.exists("spindens.cube"):
            shutil.copy(
                "spindens.cube",
                f"../output_densities/{mol.name}_{charge}_{spin}_{method}_{calc}.spindens.cube",
            )

        if os.path.exists("spindensity.cub"):
            shutil.copy(
                "spindensity.cub",
                f"../output_densities/{mol.name}_{charge}_{spin}_{method}_{calc}.spindens.cube",
            )


def save_dftb_trajectory(output_prefix):
    """Saves the geo_end.xyz and md.out files to a temporary directory where an MDTrajectory
    object can go read the data it needs.

    Parameters
    ----------
    output_prefix : str
        name of the output trajectory files prefix
    """

    os.makedirs("../MD_data", exist_ok=True)

    shutil.move("md.out", f"../MD_data/{output_prefix}_md.out")
    shutil.move("geo_end.xyz", f"../MD_data/{output_prefix}_geo_end.xyz")


def clean_suffix(source: str) -> str:
    """
    Removes characters not compatible with a file name from the given string

    Arguments
    ---------
    source: str
        The string to be parsed for cleanup
    
    Returns
    -------
    str
        The cleaned up string
    """

    string = deepcopy(source)

    for char in ["\\", "/", ",", "*", "(", ")", "[", "]"]:
        string = string.replace(char, "-")

    return string