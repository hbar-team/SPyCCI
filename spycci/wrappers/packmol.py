import os, shutil, sh
import logging
from tempfile import mkdtemp

from spycci.constants import atomic_masses, avogadro
from spycci.systems import System
from spycci.tools import process_output

logger = logging.getLogger(__name__)


def packmol_cube(
    solute: System,
    solvent: System,
    nsolv: int = None,
    target_dens: float = None,
    cube_side: float = None,
):
    """Interface for running packmol to generate a cubic solvation box.

    Parameters
    ----------
    solute : str
        System object of the solute molecule
    solvent : str
        System object of the solvent molecule

    # two of the following three parameters are required, the third will be calculated
    # based on the other two
    nsolv : int, optional
        number of solvent molecules to add in the box
    target_dens : float, optional
        target density for the solvated box, in g/L
    cube_side : float, optional
        length of the box side, in Å

    Returns
    -------
    Returns a System object
    """

    if nsolv and target_dens and cube_side:
        logger.error(
            "At least one of ( nsolv | target_dens | cube_side ) must be left out."
        )
        return None

    elif nsolv and target_dens:

        solvent_grams = (
            sum([atomic_masses[atom] for atom, _ in solvent.geometry]) * nsolv / avogadro
        )  # g
        solute_grams = (
            sum([atomic_masses[atom] for atom, _ in solute.geometry]) / avogadro
        )  # g

        target_volume = (solvent_grams + solute_grams) / target_dens  # L

        cube_side = (target_volume / 1e-27) ** (1.0 / 3)

    elif nsolv and cube_side:

        solvent_grams = (
            sum([atomic_masses[atom] for atom, _ in solvent.geometry]) * nsolv / avogadro
        )  # g
        solute_grams = (
            sum([atomic_masses[atom] for atom, _ in solute.geometry]) / avogadro
        )  # g

        volume = (cube_side ** 3) * 1e-27  # L

        target_dens = (solvent_grams + solute_grams) / volume

    elif target_dens and cube_side:

        volume = (cube_side ** 3) * 1e-27  # L

        solute_grams = (
            sum([atomic_masses[atom] for atom, _ in solute.geometry]) / avogadro
        )  # g

        target_solv_weight = (target_dens * volume) - solute_grams  # g

        nsolv = round(
            target_solv_weight
            * avogadro
            / sum([atomic_masses[atom] for atom, _ in solvent.geometry])
        )  # n° of molecules

        solvent_grams = (
            sum([atomic_masses[atom] for atom, _ in solvent.geometry]) * nsolv / avogadro
        )  # g

        # recalculating density with actual number of solvent molecules
        target_dens = (solvent_grams + solute_grams) / volume

    else:
        logger.error(
            "At least two of ( nsolv | target_dens | cube_side ) must be provided."
        )
        return None

    logger.info(f"{solute.name} - Generating solvation box with {nsolv} {solvent.name}s")
    # Also print EFFECTIVE DENSITY!
    logger.debug(
        f"Packmol solvated {solute.name} - cubic box with {nsolv} {solvent.name} molecules, side {cube_side} Å, density {target_dens} g/L."
    )

    tdir = mkdtemp(
        prefix=f"{solute.name}_{nsolv}{solvent.name}s" + "_",
        suffix=f"_packmol",
        dir=os.getcwd(),
    )

    with sh.pushd(tdir):

        solute.geometry.write_xyz(f"{solute.name}.xyz")
        solvent.geometry.write_xyz(f"{solvent.name}.xyz")

        os.system(f"obabel {solute.name}.xyz -O {solute.name}.pdb")
        os.system(f"obabel {solvent.name}.xyz -O {solvent.name}.pdb")

        with open("input.inp", "w") as f:
            f.write(
                "tolerance 2.0\n"
                f"filetype pdb\n\n"
                f"output {solute.name}_{nsolv}{solvent.name}s.pdb\n\n"
                f"structure {solute.name}.pdb\n"
                "  number 1\n"
                "  resnumbers 3\n"
                "  center\n"
                f"  fixed {(cube_side)/2} {(cube_side)/2} {(cube_side)/2} 0. 0. 0.\n"
                "end structure\n"
                f"structure {solvent.name}.pdb\n"
                f"  number {nsolv}\n"
                "  resnumbers 3\n"
                f"  inside cube 1. 1. 1. {cube_side-1}\n"
                "end structure\n\n"
            )

        os.system("packmol < input.inp > output.out")

        os.system(
            f"obabel {solute.name}_{nsolv}{solvent.name}s.pdb -xyz -O {solute.name}_{nsolv}{solvent.name}s.xyz"
        )

        solvated_molecule = System.from_xyz(
            f"{solute.name}_{nsolv}{solvent.name}s.xyz", box_side=cube_side,
        )

        process_output(mol=solute, method="packmol", calc=f"{nsolv}{solvent.name}_cube")

        # saving packmol files
        os.makedirs("../packmol_files", exist_ok=True)
        shutil.copy(
            f"{solute.name}_{nsolv}{solvent.name}s.pdb",
            f"../packmol_files/{solute.name}_{nsolv}{solvent.name}s.pdb",
        )
        with open(f"../packmol_files/{solute.name}_{nsolv}{solvent.name}s.pbc", "w") as f:
            f.write(str(solvated_molecule.box_side))

        shutil.rmtree(tdir)

    return solvated_molecule
