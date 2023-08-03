import pytest

from spycci.engines.xtb import XtbInput
from spycci.engines.orca import OrcaInput
from spycci.systems import System, Ensemble
from spycci.functions.pka import calculate_pka, auto_calculate_pka
from os.path import dirname, abspath
from shutil import rmtree

import numpy as np
from numpy.testing import assert_almost_equal

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))


def test_calculate_pka_xtb():

    protonated = System(f"{TEST_DIR}/utils/xyz_files/acetic_acid.xyz", charge=0, spin=1)
    deprotonated = System(f"{TEST_DIR}/utils/xyz_files/deprotonated_acetic_acid.xyz", charge=-1, spin=1)

    xtb = XtbInput(solvent="water")
    xtb.opt(protonated, inplace=True)
    xtb.opt(deprotonated, inplace=True)

    try:
        pka = calculate_pka(protonated, deprotonated)

    except:
        assert False, "Unexpected exception raised during pka calculation"

    assert_almost_equal(pka, 8.388911127439894, decimal=6)
    assert_almost_equal(protonated.properties.pka, 8.388911127439894, decimal=6)

    rmtree("output_files")
    rmtree("error_files")



def test_auto_calculate_pka_xtb():

    protonated = System(f"{TEST_DIR}/utils/xyz_files/acetic_acid.xyz")
    xtb = XtbInput(solvent="water")

    try:
        pka, _ = auto_calculate_pka(
            protonated,
            method_el=xtb,
            method_vib=xtb,
            method_opt=xtb,
            ncores=2,
            maxcore=2000,
        )

    except:
        assert False, "Unexpected exception raised during pka calculation"

    assert_almost_equal(pka, 8.219089610928382, decimal=6)
    assert_almost_equal(protonated.properties.pka, 8.219089610928382, decimal=6)

    rmtree("output_files")
    rmtree("error_files")