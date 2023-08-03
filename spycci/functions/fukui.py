from os.path import join
from copy import deepcopy
from typing import List, Dict, Union
from enum import Enum

from spycci.systems import System
from spycci.engines.orca import OrcaInput
from spycci.engines.xtb import XtbInput
from spycci.tools.cubetools import Cube


FUKUI_CUBE_WARNING = "Fukui function cube file, the atomic charges column contains the localized Mulliken-charge-based fukui values"

class CubeGrids(Enum):
    """
    Grids to be used by the engines in generating cubes. The first value represents the
    homogeneous spacing used by orca while the second the grid step used by xTB.
    """
    ULTRAFINE = [400, 0.1]
    FINE = [200, 0.2]
    NORMAL = [100, 0.4]
    COARSE = [50, 0.8]


def calculate_fukui(
    molecule: System,
    engine: Union[OrcaInput, XtbInput],
    spins_states: Union[None, List[int]] = None,
    cube_grid: Union[CubeGrids, None] = CubeGrids.NORMAL,
    ncores: int = None,
    maxcore: int = 1000,
) -> None:
    """
    Computes the Fukui f+, f- and f0 functions starting from a given input molecule. The
    functions are saved in Gaussian cube compatible format and stored in the 
    `output_densities` folder. Localized Fukui functions are also computed from the Mulliken
    and Hirshfeld charges and saved in the molecule properties. Please notice that the charges
    in the `.fukui.cube` produced are replaced by the Mulliken condensed fukui functions.

    Parameters
    ----------
    molecule: System
        The System object containing the geometry of the selected molecule (if the geometry
        has not been optimized, please enable the optimize option)
    orca: Engine
        The engine object that defines the protocol to be used in the calculation.
    spin_states: Union[None, List[int]]
        If set to None, when adding or subtracting electrons will automatically switch the
        spin state from singlet to doublet and vice versa (Maximum one unpaired electrons).
        If manually set to `List[int]` will force the spin multeplicities according to the
        user specified values. The order of the spin multiplicity values is: molecule with
        one electron added (-1), the molecule as it is (0) and the molecule  with one of its
        electrons removed (+1).
    cube_grid: Union[CubeGrids, None]
        Resolution for the cube files (default Normal). If set to None the function will avoid
        the generation of cube files and will only compute condensed fukui values.
    ncores : int, optional
        The number of cores, by default all available cores (None)
    maxcore: int
        The maximum amount of memory in MB to be allocated for each core.
    """

    if type(engine) not in [OrcaInput, XtbInput]:
        raise TypeError("calculate_fukui currently only supports xTB and Orca")

    save_cubes = False if cube_grid is None else True
    
    # RUN ALL THE REQUIRED CALCULATIONS
    # --------------------------------------------------------------------------------------
    # Compute a single point for the molecule
    if type(engine) == OrcaInput:
        engine.spe(
            molecule,
            save_cubes=save_cubes,
            cube_dim=cube_grid.value[0] if save_cubes else 0,
            hirshfeld=True,
            inplace=True,
            ncores=ncores,
            maxcore=maxcore,
        )

    elif type(engine) == XtbInput:
        engine.spe(
            molecule,
            save_cubes=save_cubes,
            cube_step=cube_grid.value[1] if save_cubes else 0,
            inplace=True,
            ncores=ncores,
        )

    # Compute a single point for the molecule with the addition of one electron.
    cation = deepcopy(molecule)
    cation.charge += 1

    if spins_states is not None:
        cation.spin = spins_states[2]
    else:
        cation.spin = 1 if molecule.spin == 2 else 2

    if type(engine) == OrcaInput:
        engine.spe(
            cation,
            save_cubes=save_cubes,
            cube_dim=cube_grid.value[0] if save_cubes else 0,
            hirshfeld=True,
            inplace=True,
            ncores=ncores,
            maxcore=maxcore
        )

    elif type(engine) == XtbInput:
        engine.spe(
            cation,
            save_cubes=save_cubes,
            cube_step=cube_grid.value[1] if save_cubes else 0,
            inplace=True,
            ncores=ncores,
        )

    # Compute a single point for the molecule with the subtraction of one electron.
    anion = deepcopy(molecule)
    anion.charge -= 1

    if spins_states is not None:
        anion.spin = spins_states[0]
    else:
        anion.spin = 1 if molecule.spin == 2 else 2

    if type(engine) == OrcaInput:
        engine.spe(
            anion,
            save_cubes=save_cubes,
            cube_dim=cube_grid.value[0] if save_cubes else 0,
            hirshfeld=True,
            inplace=True,
            ncores=ncores,
            maxcore=maxcore
        )

    elif type(engine) == XtbInput:
        engine.spe(
            anion,
            save_cubes=save_cubes,
            cube_step=cube_grid.value[1] if save_cubes else 0,
            inplace=True,
            ncores=ncores,
        )

    # COMPUTE CONDENSED FUKUI FUNCTIONS
    #---------------------------------------------------------------------------------------
    # Compute the localized fukui function values from the Mulliken charges
    localized_fukui_mulliken = {"f+": [], "f-": [], "f0": []}
    for atom in range(molecule.geometry.atomcount):
        localized_fukui_mulliken["f+"].append(
            -(anion.properties.mulliken_charges[atom] - molecule.properties.mulliken_charges[atom])
        )
        localized_fukui_mulliken["f-"].append(
            -(molecule.properties.mulliken_charges[atom] - cation.properties.mulliken_charges[atom])
        )
        localized_fukui_mulliken["f0"].append(
            -(anion.properties.mulliken_charges[atom] - cation.properties.mulliken_charges[atom]) / 2
        )
    
    molecule.properties.set_condensed_fukui_mulliken(localized_fukui_mulliken, engine)

    # Compute the localized fukui function values from the Hirshfeld charges (orca only)
    if type(engine) == OrcaInput:

        localized_fukui_hirshfeld = {"f+": [], "f-": [], "f0": []}

        for atom in range(molecule.geometry.atomcount):
            localized_fukui_hirshfeld["f+"].append(
                -(anion.properties.hirshfeld_charges[atom] - molecule.properties.hirshfeld_charges[atom])
            )
            localized_fukui_hirshfeld["f-"].append(
                -(molecule.properties.hirshfeld_charges[atom] - cation.properties.hirshfeld_charges[atom])
            )
            localized_fukui_hirshfeld["f0"].append(
                -(anion.properties.hirshfeld_charges[atom] - cation.properties.hirshfeld_charges[atom]) / 2
            )
        
        molecule.properties.set_condensed_fukui_hirshfeld(localized_fukui_hirshfeld, engine)

    # LOAD THE VOLUMETRIC FILES AND COMPUTE THE FUKUI FUNCTIONS
    # --------------------------------------------------------------------------------------
    # Load cubes from the output_densities folder
    if save_cubes:

        cubes: Dict[int, Cube] = {}
        for mol in [cation, molecule, anion]:
            cubename = f"{mol.name}_{mol.charge}_{mol.spin}_{engine.output_suffix}_spe.eldens.cube"
            cube = Cube.from_file(join("./output_densities", cubename))
            cubes[mol.charge] = cube

        # Check if all the densities have been loaded correctly
        if len(cubes) != 3:
            raise RuntimeError(f"Three cube files expected, {len(cubes)} found.")

        # Compute the f+ Fukui function
        f_plus = cubes[anion.charge] - cubes[molecule.charge]
        f_plus.charges = localized_fukui_mulliken["f+"]
        f_plus.save(
            join("./output_densities", f"{molecule.name}_{engine.output_suffix}_Fukui_plus.fukui.cube"),
            comment_1st=FUKUI_CUBE_WARNING,
            comment_2nd="Fukui f+",
        )

        # Compute the f- Fukui function
        f_minus = cubes[molecule.charge] - cubes[cation.charge]
        f_minus.charges = localized_fukui_mulliken["f-"]
        f_minus.save(
            join("./output_densities", f"{molecule.name}_{engine.output_suffix}_Fukui_minus.fukui.cube"),
            comment_1st=FUKUI_CUBE_WARNING,
            comment_2nd="Fukui f-",
        )

        # Compute the f0 Fukui function
        f_zero = (cubes[anion.charge] - cubes[cation.charge]).scale(0.5)
        f_zero.charges = localized_fukui_mulliken["f0"]
        f_zero.save(
            join("./output_densities", f"{molecule.name}_{engine.output_suffix}_Fukui_zero.fukui.cube"),
            comment_1st=FUKUI_CUBE_WARNING,
            comment_2nd="Fukui f0",
        )
    