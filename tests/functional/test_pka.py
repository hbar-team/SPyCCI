import pytest

from spycci.engines.xtb import XtbInput
from spycci.engines.orca import OrcaInput
from spycci.systems import System, Ensemble
from spycci.functions.pka import (
    calculate_pka,
    calculate_pka_oxonium_scheme,
    auto_calculate_pka,
    run_pka_workflow,
)
from os.path import dirname, abspath
from shutil import rmtree

import numpy as np
from numpy.testing import assert_almost_equal

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))


def test_calculate_pka_xtb():

    protonated = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/acetic_acid.xyz", charge=0, spin=1)
    deprotonated = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/deprotonated_acetic_acid.xyz", charge=-1, spin=1)

    xtb = XtbInput(solvent="water")
    xtb.opt(protonated, inplace=True)
    xtb.opt(deprotonated, inplace=True)

    try:
        pka = calculate_pka(protonated, deprotonated)

    except:
        assert False, "Unexpected exception raised during pka calculation"

    # NOTE: xtb 6.7.1 gives 8.40, xtb 6.6.1 gives 8.39
    # I set the accuracy to the 2nd decimal - Luca
    assert_almost_equal(pka.direct, 8.401564242900081, decimal=2)
    assert_almost_equal(protonated.properties.pka.direct, 8.401564242900081, decimal=2)

    assert pka.oxonium is None
    assert pka.oxonium_cosmors is None

    rmtree("output_files")
    rmtree("error_files")


def test_calculate_pka_oxonium_scheme_xtb():

    protonated = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/acetic_acid.xyz", charge=0, spin=1)
    deprotonated = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/deprotonated_acetic_acid.xyz", charge=-1, spin=1)
    water = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=0, spin=1)
    oxonium = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/oxonium.xyz", charge=1, spin=1)

    xtb = XtbInput(solvent="water")
    xtb.opt(protonated, inplace=True)
    xtb.opt(deprotonated, inplace=True)
    xtb.opt(water, inplace=True)
    xtb.opt(oxonium, inplace=True)

    try:
        pka = calculate_pka_oxonium_scheme(protonated, deprotonated, water, oxonium)

    except:
        assert False, "Unexpected exception raised during pka calculation"

    # NOTE: xtb 6.7.1 gives 13.22, xtb 6.6.1 gives 13.21
    # I set the accuracy to the 2nd decimal - Luca
    assert_almost_equal(pka.oxonium, 13.22416853109027, decimal=2)
    assert_almost_equal(protonated.properties.pka.oxonium, 13.22416853109027, decimal=2)

    assert pka.direct is None
    assert pka.oxonium_cosmors is None

    rmtree("output_files")
    rmtree("error_files")


def test_auto_calculate_pka_xtb():

    protonated = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/acetic_acid.xyz")
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

    # NOTE: xtb 6.7.1 gives 8.30, xtb 6.6.1 gives 8.22
    # I set the accuracy to the 1st decimal - Luca
    assert_almost_equal(pka.direct, 8.306140678635513, decimal=1)
    assert_almost_equal(protonated.properties.pka.direct, 8.306140678635513, decimal=1)

    assert pka.oxonium is None
    assert pka.oxonium_cosmors is None

    rmtree("output_files")
    rmtree("error_files")


# =================================================================
#     The following tests have been developed for ORCA 6.1.0
# =================================================================


def test_run_pka_workflow_different_geometry():

    protonated = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/acetic_acid.xyz", charge=0, spin=1)
    deprotonated = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/deprotonated_acetic_acid.xyz", charge=-1, spin=1)

    xtb = XtbInput(solvent="water")
    orca = OrcaInput(method="BP86", basis_set="def2-TZVPD", solvent="water")

    try:
        pka, opt_system = run_pka_workflow(
            protonated,
            deprotonated,
            orca,
            method_geometry=xtb,
            use_cosmors=True,
            ncores=4,
        )

    except:
        assert False, "Unexpected exception raised during pka calculation"

    expected_pka = {
        "direct": 7.42913365186978,
        "oxonium": 18.423841456788455,
        "oxonium COSMO-RS": 0.2627522624964962,
    }

    expected_free_energies = {
        "G(solv) Protonated": -229.169292065185,
        "G(solv) Deprotonated": -228.722399358581,
        "G(solv) Water": -76.476129950296,
        "G(solv) Oxonium": -76.879177832864,
        "dG(COSMO-RS) Protonated": -0.011667116259,
        "dG(COSMO-RS) Deprotonated": -0.133820497721,
        "dG(COSMO-RS) Water": -0.012171743587,
        "dG(COSMO-RS) Oxonium": -0.172502808099,
        "G(vac) Protonated": -229.15908478666202,
        "G(vac) Deprotonated": -228.61156082994202,
        "G(vac) Water": -76.46491195712699,
        "G(vac) Oxonium": -76.725590646525,
    }

    # NOTE: xtb 6.7.1 and xtb 6.6.1 results differ at the 2nd decimal unit - Luca
    for key, value in expected_pka.items():
        assert_almost_equal(pka[key], value, decimal=2)
        assert_almost_equal(opt_system.properties.pka[key], value, decimal=2)

    for key, value in expected_free_energies.items():
        assert_almost_equal(pka.free_energies[key], value, decimal=2)
        assert_almost_equal(opt_system.properties.pka.free_energies[key], value, decimal=2)

    # for key in pka.keys():
    #     if key not in expected_pka.keys():
    #         assert False, "Unexpected key found in pka dictionary"

    # for key in pka.free_energies.keys():
    #     if key not in expected_free_energies.keys():
    #         assert False, "Unexpected key found in free energies dictionary"

    rmtree("output_files")
    rmtree("error_files")


def test_run_pka_workflow_different_electronic():

    protonated = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/acetic_acid.xyz", charge=0, spin=1)
    deprotonated = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/deprotonated_acetic_acid.xyz", charge=-1, spin=1)

    xtb = XtbInput(solvent="water")
    orca = OrcaInput(method="BP86", basis_set="def2-TZVPD", solvent="water")

    try:
        pka, opt_system = run_pka_workflow(
            protonated,
            deprotonated,
            method_vibrational=xtb,
            method_electonic=orca,
            method_geometry=xtb,
            use_cosmors=True,
            ncores=4,
        )

    except:
        assert False, "Unexpected exception raised during pka calculation"

    expected_pka = {
        "direct": 7.8319719542137785,
        "oxonium": 18.420967059853076,
        "oxonium COSMO-RS": 0.4592855502229256,
    }

    expected_free_energies = {
        'G(solv) Protonated': -229.171070654057,
        'G(solv) Deprotonated': -228.723302137173,
        'G(solv) Water': -76.47731686852,
        'G(solv) Oxonium': -76.881246810591,
        'dG(COSMO-RS) Protonated': -0.011667116259,
        'dG(COSMO-RS) Deprotonated': -0.133820497721,
        'dG(COSMO-RS) Water': -0.012171743587,
        'dG(COSMO-RS) Oxonium': -0.172502808099,
        'G(vac) Protonated': -229.1605913353,
        'G(vac) Deprotonated': -228.61180103379402,
        'G(vac) Water': -76.46602068902699,
        'G(vac) Oxonium': -76.727538440421
    }

    # NOTE: xtb 6.7.1 and xtb 6.6.1 results differ at the 2nd decimal unit - Luca
    for key, value in expected_pka.items():
        assert_almost_equal(pka[key], value, decimal=2)
        assert_almost_equal(opt_system.properties.pka[key], value, decimal=2)

    for key, value in expected_free_energies.items():
        assert_almost_equal(pka.free_energies[key], value, decimal=2)
        assert_almost_equal(opt_system.properties.pka.free_energies[key], value, decimal=2)

    # for key in pka.keys():
    #     if key not in expected_pka.keys():
    #         assert False, "Unexpected key found in pka dictionary"

    # for key in pka.free_energies.keys():
    #     if key not in expected_free_energies.keys():
    #         assert False, "Unexpected key found in free energies dictionary"

    rmtree("output_files")
    rmtree("error_files")
