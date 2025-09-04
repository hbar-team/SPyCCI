import pytest

from spycci.engines.xtb import XtbInput
from spycci.engines.orca import OrcaInput
from spycci.systems import System, Ensemble
from spycci.functions.fukui import calculate_fukui, CubeGrids
from os.path import dirname, abspath
from numpy.testing import assert_array_almost_equal
from shutil import rmtree
from spycci.core.dependency_finder import find_orca_version

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))


# =================================================================
#     The following tests have been developed for ORCA 6.0.1
# =================================================================


@pytest.mark.skipif(find_orca_version() != "6.0.1", reason="Test designed for orca==6.0.1")
def test_calculate_fukui():

    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=0, spin=1)
    orca = OrcaInput("PBE", basis_set="def2-SVP", solvent="water")

    try:
        calculate_fukui(mol, orca, cube_grid=CubeGrids.COARSE, ncores=2)
    except:
        assert False, "Exception occurred on calculate_fukui with COARSE grid"

    assert_array_almost_equal(
        mol.properties.condensed_fukui_mulliken["f0"],
        [0.3151970, 0.3419340, 0.3428690],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_mulliken["f+"],
        [-0.1230240, 0.5605780, 0.5624460],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_mulliken["f-"],
        [0.7534180, 0.1232900, 0.1232920],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_hirshfeld["f0"],
        [0.4735585, 0.2629585, 0.2634750],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_hirshfeld["f+"],
        [0.2595910, 0.3696890, 0.3707040],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_hirshfeld["f-"],
        [0.6875260, 0.1562280, 0.1562460],
        decimal=6,
    )

    rmtree("output_files")
    rmtree("output_densities")
    rmtree("error_files", ignore_errors=True)


@pytest.mark.skipif(find_orca_version() != "6.0.1", reason="Test designed for orca==6.0.1")
def test_calculate_fukui_no_cube():

    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=0, spin=1)
    orca = OrcaInput("PBE", basis_set="def2-SVP", solvent="water")

    try:
        calculate_fukui(mol, orca, cube_grid=None, ncores=2)
    except:
        assert False, "Exception occurred on calculate_fukui with COARSE grid"

    assert_array_almost_equal(
        mol.properties.condensed_fukui_mulliken["f0"],
        [0.3151970, 0.3419340, 0.3428690],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_mulliken["f+"],
        [-0.1230240, 0.5605780, 0.5624460],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_mulliken["f-"],
        [0.7534180, 0.1232900, 0.1232920],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_hirshfeld["f0"],
        [0.4735585, 0.2629585, 0.2634750],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_hirshfeld["f+"],
        [0.2595910, 0.3696890, 0.3707040],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_hirshfeld["f-"],
        [0.6875260, 0.1562280, 0.1562460],
        decimal=6,
    )

    rmtree("output_files")
    rmtree("error_files", ignore_errors=True)
