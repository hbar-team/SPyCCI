import pytest

from spycci.engines.xtb import XtbInput
from spycci.engines.orca import OrcaInput
from spycci.systems import System, Ensemble
from spycci.functions.potential import calculate_reduction_potential
from os.path import dirname, abspath
from shutil import rmtree

import numpy as np
from numpy.testing import assert_almost_equal

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))

# =================================================================
#      The following test has been developed for XTB 6.7.0
# =================================================================

def test_calculate_potential_xtb():

    reduced = System.from_xyz(
        f"{TEST_DIR}/utils/xyz_files/1,4-dimethoxybenzene.xyz", charge=0, spin=1
    )
    oxidised = System.from_xyz(
        f"{TEST_DIR}/utils/xyz_files/1,4-dimethoxybenzene.xyz", charge=1, spin=2
    )

    xtb = XtbInput(solvent="water")
    xtb.opt(oxidised, inplace=True)
    xtb.opt(reduced, inplace=True)

    try:
        potential, exc_electrons, exc_protons = calculate_reduction_potential(
            oxidised, reduced
        )

    except:
        assert False, "Unexpected exception raised during reduction potential calculation"

    assert exc_electrons == 1
    assert exc_protons == 0
    assert_almost_equal(potential, 1.2977739, decimal=4)

    rmtree("output_files")
    rmtree("error_files")


def test_calculate_potential_xtb_pcet():
    reduced = System.from_xyz(
        f"{TEST_DIR}/utils/xyz_files/2-methoxyphenol_prot.xyz", charge=0, spin=1
    )
    oxidised = System.from_xyz(
        f"{TEST_DIR}/utils/xyz_files/2-methoxyphenol_deprot.xyz", charge=0, spin=2
    )

    xtb = XtbInput(solvent="water")
    xtb.opt(oxidised, inplace=True)
    xtb.opt(reduced, inplace=True)

    try:
        potential, exc_electrons, exc_protons = calculate_reduction_potential(
            oxidised, reduced
        )

    except:
        assert False, "Unexpected exception raised during reduction potential calculation"

    assert exc_electrons == 1
    assert exc_protons == 1
    assert_almost_equal(potential, 0.8602223761724688, decimal=3)

    rmtree("output_files")
    rmtree("error_files")
