import pytest

from spycci.engines.xtb import XtbInput
from spycci.engines.orca import OrcaInput
from spycci.systems import System, Ensemble
from spycci.functions.fukui import calculate_fukui, CubeGrids
from os.path import dirname, abspath
from numpy.testing import assert_array_almost_equal
from shutil import rmtree

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))


def test_calculate_fukui():

    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=0, spin=1)
    orca = OrcaInput("PBE", basis_set="def2-SVP", solvent="water")

    try:
        calculate_fukui(mol, orca, cube_grid=CubeGrids.COARSE, ncores=2)
    except:
        assert False, "Exception occurred on calculate_fukui with COARSE grid"

    assert_array_almost_equal(
        mol.properties.condensed_fukui_mulliken["f0"],
        [0.315185, 0.3419, 0.342916],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_mulliken["f+"],
        [-0.123065, 0.560517, 0.562548],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_mulliken["f-"],
        [0.753435, 0.123282, 0.123284],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_hirshfeld["f0"],
        [0.473557, 0.262937, 0.263497],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_hirshfeld["f+"],
        [0.259586, 0.369647, 0.370749],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_hirshfeld["f-"],
        [0.687529, 0.156227, 0.156244],
        decimal=6,
    )

    rmtree("output_files")
    rmtree("output_densities")
    rmtree("error_files", ignore_errors=True)


def test_calculate_fukui_no_cube():

    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=0, spin=1)
    orca = OrcaInput("PBE", basis_set="def2-SVP", solvent="water")

    try:
        calculate_fukui(mol, orca, cube_grid=None, ncores=2)
    except:
        assert False, "Exception occurred on calculate_fukui with COARSE grid"

    assert_array_almost_equal(
        mol.properties.condensed_fukui_mulliken["f0"],
        [0.315185, 0.3419, 0.342916],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_mulliken["f+"],
        [-0.123065, 0.560517, 0.562548],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_mulliken["f-"],
        [0.753435, 0.123282, 0.123284],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_hirshfeld["f0"],
        [0.473557, 0.262937, 0.263497],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_hirshfeld["f+"],
        [0.259586, 0.369647, 0.370749],
        decimal=6,
    )

    assert_array_almost_equal(
        mol.properties.condensed_fukui_hirshfeld["f-"],
        [0.687529, 0.156227, 0.156244],
        decimal=6,
    )

    rmtree("output_files")
    rmtree("error_files", ignore_errors=True)
