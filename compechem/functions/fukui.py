from os import listdir
from os.path import join
from copy import deepcopy
from typing import List, Dict, Union

from compechem.systems import System
from compechem.wrappers.orca import OrcaInput
from compechem.tools.cubetools import Cube


def compute_fukui_densities(
    molecule: System,
    orca: OrcaInput,
    spins_states: Union[None, List[int]] = None,
    optimize: bool = False,
    maxcore: int = 1000,
) -> None:
    """
    Computes the Fukui f+, f- and f0 functions starting from a given input molecule. The
    functions are saved in Gaussian cube format and stored in the `output_densities` folder.

    Parameters
    ----------
    molecule: System
        The System object containing the geometry of the selected molecule (if the geometry
        has not been optimized, please enable the optimize option)
    orca: OrcaInput
        The orca input wrapper object that defines the protocol to be used in the calculation.
    spin_states: Union[None, List[int]]
        If set to None, when adding or subtracting electrons will automatically switch the
        spin state from singlet to doublet and vice versa (Maximum one unpaired electrons).
        If manually set to `List[int]` will force the spin multeplicities according to the
        user specified values. The order of the spin multiplicity values is: molecule with
        one electron added (-1), the molecule as it is (0) and the molecule  with one of its
        electrons removed (+1).
    optimize: bool
        If set to True, it will run the optimization of the origin system at the same level
        of theory specified by the method option.
    maxcore: int
        The maximum amount of memory in MB to be allocated for each core.
    """
    # Make a copy of the oriCompute a single point for the original molecule
    origin = deepcopy(molecule)
    if optimize:
        orca.opt(origin, save_cubes=True, inplace=True, maxcore=maxcore)
    else:
        orca.spe(origin, save_cubes=True, inplace=True, maxcore=maxcore)

    # Compute a single point for the molecule with the addition of one electron.
    cation = deepcopy(origin)
    cation.charge += 1

    if spins_states is not None:
        cation.spin = spins_states[2]
    else:
        cation.spin = 1 if origin.spin == 2 else 2

    orca.spe(cation, save_cubes=True, inplace=True, maxcore=maxcore)

    # Compute a single point for the molecule with the subtraction of one electron.
    anion = deepcopy(origin)
    anion.charge -= 1

    if spins_states is not None:
        anion.spin = spins_states[0]
    else:
        anion.spin = 1 if origin.spin == 2 else 2

    orca.spe(anion, save_cubes=True, inplace=True, maxcore=maxcore)

    # Load cubes from the output_densities folder
    cubes: Dict[int, Cube] = {}
    for file in listdir("./output_densities"):
        if file.endswith("eldens.cube"):
            cube = Cube.from_file(join("./output_densities", file))
            current_charge = int(file.split("_")[1])
            cubes[current_charge] = cube

    # Check if all the densities have been loaded correctly
    if len(cubes) != 3:
        raise RuntimeError(f"Three cube files expected, {len(cubes)} found.")

    # Compute the f+ Fukui function
    f_plus = cubes[anion.charge] - cubes[origin.charge]
    f_plus.save(
        join("./output_densities", f"{molecule.name}_Fukui_plus.cube"), comment="Fukui f+"
    )

    # Compute the f- Fukui function
    f_minus = cubes[origin.charge] - cubes[cation.charge]
    f_minus.save(
        join("./output_densities", f"{molecule.name}_Fukui_minus.cube"), comment="Fukui f-"
    )

    # Compute the f0 Fukui function
    f_zero = (cubes[anion.charge] - cubes[cation.charge]).scale(0.5)
    f_zero.save(
        join("./output_densities", f"{molecule.name}_Fukui_zero.cube"), comment="Fukui f0"
    )