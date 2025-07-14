import pytest

from spycci.engines.xtb import XtbInput
from spycci.systems import System

from os import listdir
from os.path import dirname, abspath
from shutil import rmtree

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

# =================================================================
#      The following test has been developed for XTB 6.7.1
# =================================================================

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))

# Test the XtbInput class constructor
def test_XtbInput___init__():

    try:
        engine = XtbInput(solvent="water")

    except:
        assert False, "Unenxpected exception raised during XtbInput class construction"

    else:
        assert engine.method == "gfn2"
        assert engine.level_of_theory == "XtbInput || method: gfn2 | solvent: water"


# Test the spe() function on a radical cation water molecule in DMSO
def test_XtbInput_spe():

    engine = XtbInput(solvent="DMSO")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=1, spin=2)

    try:
        engine.spe(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during SPE calculation"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert_array_almost_equal(
            mol.properties.electronic_energy, -4.547249570099, decimal=6
        )

        expected_mulliken_charges = np.array([-0.121, 0.560, 0.560])
        assert_array_almost_equal(
            expected_mulliken_charges, mol.properties.mulliken_charges, decimal=3
        )
        expected_mulliken_spin_populations = np.array([1.00, 0.00, 0.00])
        assert_array_almost_equal(
            expected_mulliken_spin_populations,
            mol.properties.mulliken_spin_populations,
            decimal=4,
        )

        rmtree("output_files")
        rmtree("error_files")


# Test the spe() function on a radical cation water molecule in DMSO with no inplace
def test_XtbInput_spe_no_inplace():

    engine = XtbInput(solvent="DMSO")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=1, spin=2)

    try:
        newmol = engine.spe(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during SPE calculation"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert_array_almost_equal(
            newmol.properties.electronic_energy, -4.547249570099, decimal=6
        )

        expected_mulliken_charges = np.array([-0.121, 0.560, 0.560])
        assert_array_almost_equal(
            expected_mulliken_charges, newmol.properties.mulliken_charges, decimal=3
        )
        expected_mulliken_spin_populations = np.array([1.00, 0.00, 0.00])
        assert_array_almost_equal(
            expected_mulliken_spin_populations,
            newmol.properties.mulliken_spin_populations,
            decimal=4,
        )

        rmtree("output_files")
        rmtree("error_files")


# Test the opt() function on a urea molecule in vacuum
def test_XtbInput_opt():

    engine = XtbInput(solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/urea.xyz")

    try:
        engine.opt(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -14.097142459981, decimal=6)
        assert_almost_equal(mol.properties.free_energy_correction, 0.032427777313, decimal=4)
        assert_almost_equal(mol.properties.gibbs_free_energy, -14.064714682668, decimal=4)

        expected_mulliken_charges = np.array(
            [0.345, -0.301, 0.192, 0.170, -0.300, 0.170, 0.192, -0.470]
        )

        assert_array_almost_equal(
            expected_mulliken_charges, mol.properties.mulliken_charges, decimal=3
        )

        expected_geometry = [
            np.array([0.38572247721477, 4.16024211476103, 3.49696517842814]),
            np.array([0.49249877420667, 3.54386302158299, 2.28280472105639]),
            np.array([0.47349293185483, 2.54024248272376, 2.27028654511170]),
            np.array([0.78255213235431, 4.02888732249512, 1.45441390653355]),
            np.array([0.49592224672893, 5.52076774691656, 3.45053768495885]),
            np.array([0.78395028923093, 6.01190423416108, 2.62507311147151]),
            np.array([0.47712970895806, 6.01670697903511, 4.32319081549932]),
            np.array([0.16900193748757, 3.55540746889578, 4.52165211842273]),
        ]
        assert_array_almost_equal(expected_geometry, mol.geometry.coordinates, decimal=3)

        rmtree("output_files")
        rmtree("error_files")


# Test the opt() function on a urea molecule in vacuum with no inplace option
def test_XtbInput_opt_no_inplace():

    engine = XtbInput(solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/urea.xyz")

    try:
        newmol = engine.opt(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -14.097142459981, decimal=6)
        assert_almost_equal(newmol.properties.free_energy_correction, 0.032427777313, decimal=4)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -14.064714682668, decimal=4)

        expected_mulliken_charges = np.array(
            [0.345, -0.301, 0.192, 0.170, -0.300, 0.170, 0.192, -0.470]
        )

        assert_array_almost_equal(
            expected_mulliken_charges, newmol.properties.mulliken_charges, decimal=3
        )

        expected_geometry = [
            np.array([0.38572247721477, 4.16024211476103, 3.49696517842814]),
            np.array([0.49249877420667, 3.54386302158299, 2.28280472105639]),
            np.array([0.47349293185483, 2.54024248272376, 2.27028654511170]),
            np.array([0.78255213235431, 4.02888732249512, 1.45441390653355]),
            np.array([0.49592224672893, 5.52076774691656, 3.45053768495885]),
            np.array([0.78395028923093, 6.01190423416108, 2.62507311147151]),
            np.array([0.47712970895806, 6.01670697903511, 4.32319081549932]),
            np.array([0.16900193748757, 3.55540746889578, 4.52165211842273]),
        ]
        assert_array_almost_equal(expected_geometry, newmol.geometry.coordinates, decimal=3)

        rmtree("output_files")
        rmtree("error_files")


# Test the freq() function on a urea molecule in vacuum
def test_XtbInput_freq():

    engine = XtbInput(solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/urea.xyz")

    try:
        engine.freq(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during frequency analysis"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -14.093063923335, decimal=6)
        assert_almost_equal(mol.properties.free_energy_correction, 0.033817193430, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -14.059246729905, decimal=6)

        expected_mulliken_charges = np.array(
            [0.334, -0.301, 0.191, 0.169, -0.301, 0.169, 0.191, -0.452]
        )
        assert_array_almost_equal(
            expected_mulliken_charges, mol.properties.mulliken_charges, decimal=3
        )

        rmtree("output_files")
        rmtree("error_files")


# Test the freq() function on a urea molecule in vacuum no inplace
def test_XtbInput_freq_no_inplace():

    engine = XtbInput(solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/urea.xyz")

    try:
        newmol = engine.freq(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during frequency analysis"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -14.093063923335, decimal=6)
        assert_almost_equal(newmol.properties.free_energy_correction, 0.033817193430, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -14.059246729905, decimal=6)

        expected_mulliken_charges = np.array(
            [0.334, -0.301, 0.191, 0.169, -0.301, 0.169, 0.191, -0.452]
        )
        assert_array_almost_equal(
            expected_mulliken_charges, newmol.properties.mulliken_charges, decimal=3
        )

        rmtree("output_files")
        rmtree("error_files")


# Test the catching of runtime errors
def test_XtbInput_runtime_error_input():

    engine = XtbInput(solvent="watr")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.spe(mol, ncores=4)
    except:
        assert True
    else:
        assert False, "An exception was not raised on wrong input file."

    for filename in listdir("./"):
        if filename.endswith("_spe"):
            rmtree(filename)

def test_XtbInput_runtime_error_scf_not_converged():

    engine = XtbInput(solvent="water")
    mol = System(f"{TEST_DIR}/utils/xyz_files/cursed_water.xyz")

    try:
        engine.spe(mol, ncores=4)
    except:
        assert True
    else:
        assert False, "An exception was not raised on SCF not converged."

    for filename in listdir("./"):
        if filename.endswith("_spe"):
            rmtree(filename)