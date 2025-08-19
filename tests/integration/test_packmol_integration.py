import pytest

from spycci.wrappers import packmol
from spycci.systems import System
from os.path import dirname, abspath
from shutil import rmtree

# =================================================================
#    The following test has been developed for PACKMOL 20.14.2
# =================================================================

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))

# Test the packmol_cube() function on a urea molecule in water
def test_packmol_cube():

    solute = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/urea.xyz")
    solvent = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        solvated: System = packmol.packmol_cube(
            solute=solute, solvent=solvent, nsolv=10, cube_side=12.345
        )
    except:
        assert False, "Unexpected exception raised during generation of solvation cube"

    else:
        assert solvated.box_side == 12.345
        assert solvated.geometry.atomcount == 38

        rmtree("output_files")
        rmtree("packmol_files")
