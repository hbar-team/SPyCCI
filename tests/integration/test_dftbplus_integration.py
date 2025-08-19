import pytest

from spycci.engines.dftbplus import DFTBInput
from spycci.systems import System, Ensemble

from os import listdir
from os.path import dirname, abspath
from shutil import rmtree

import numpy as np
from numpy.testing import assert_array_almost_equal

# =================================================================
#     The following test has been developed for DFTB+ 24.1
# =================================================================

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))

# Test the DFTBInput class constructor
def test_DFTBInput___init__():

    try:
        engine = DFTBInput(parallel="mpi")

    except:
        assert False, "Unenxpected exception raised during DFTBInput class construction"

    else:
        assert engine.method == "DFTB"
        assert (
            engine.level_of_theory
            == "DFTBInput || method: DFTB | parameters: 3ob/3ob-3-1 | 3rd order: True | dispersion: False"
        )


# Test the spe() function on a water molecule in vacuum
def test_DFTBInput_spe():

    engine = DFTBInput(parallel="mpi")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.spe(mol, ncores=1, inplace=True)
    except:
        assert False, "Unexpected exception raised during SPE calculation"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert_array_almost_equal(
            mol.properties.electronic_energy, -4.0706605560, decimal=6
        )

        rmtree("output_files")
        rmtree("error_files")


# Test the spe() function on a water molecule in vacuum with no inplace option
def test_DFTBInput_spe_no_inplace():

    engine = DFTBInput(parallel="mpi")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        newmol = engine.spe(mol, ncores=1)
    except:
        assert False, "Unexpected exception raised during SPE calculation"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert_array_almost_equal(
            newmol.properties.electronic_energy, -4.0706605560, decimal=6
        )

        rmtree("output_files")
        rmtree("error_files")


# Test the opt() function on a water molecule in vacuum
def test_DFTBInput_opt():

    engine = DFTBInput(parallel="mpi")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.opt(mol, ncores=1, inplace=True)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory

        assert_array_almost_equal(
            mol.properties.electronic_energy, -4.0715923330, decimal=6
        )

        expected_geometry = [
            np.array([-3.19051453, -0.60510932, 0.00563685]),
            np.array([-2.23356419, -0.59872469, -0.00060078]),
            np.array([-3.51774330, -1.24864231, 0.63383386]),
        ]
        assert_array_almost_equal(expected_geometry, mol.geometry.coordinates, decimal=6)

        rmtree("output_files")
        rmtree("error_files")


# Test the opt() function on a water molecule in vacuum with no inplace option
def test_DFTBInput_opt_no_inplace():

    engine = DFTBInput(parallel="mpi")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        newmol = engine.opt(mol, ncores=1)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory

        assert_array_almost_equal(
            newmol.properties.electronic_energy, -4.0715923330, decimal=6
        )

        expected_geometry = [
            np.array([-3.19051453, -0.60510932, 0.00563685]),
            np.array([-2.23356419, -0.59872469, -0.00060078]),
            np.array([-3.51774330, -1.24864231, 0.63383386]),
        ]
        assert_array_almost_equal(expected_geometry, newmol.geometry.coordinates, decimal=6)

        rmtree("output_files")
        rmtree("error_files")


# Test the md_nvt() function on a water molecule in vacuum
def test_DFTBInput_md_nvt():

    engine = DFTBInput(parallel="mpi")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        ensemble: Ensemble = engine.md_nvt(mol, ncores=1, steps=100, mdrestartfreq=10)
    except:
        assert False, "Unexpected exception raised during NVT MD"

    else:
        assert len(ensemble.systems) == 11

        rmtree("output_files")
        rmtree("error_files")
        rmtree("MD_data")
        rmtree("MD_trajectories")


# Test the simulated_annealing() function on a water molecule in vacuum
def test_DFTBInput_simulated_annealing():

    engine = DFTBInput(parallel="mpi")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.simulated_annealing(
            mol,
            ncores=1,
            start_temp=1,
            target_temp=2,
            ramp_steps=5,
            hold_steps=10,
            mdrestartfreq=1,
            inplace=True,
        )
    except:
        assert False, "Unexpected exception raised during simulated annealing"

    else:
        expected_geometry = [
            np.array([-3.20710732, -0.58706447, -0.01012616]),
            np.array([-2.24817069, -0.59594656, -0.02023133]),
            np.array([-3.53500255, -1.23927885, 0.61226748]),
        ]

        # Assertion here is VERY lax due to the nature of the simulation...
        assert_array_almost_equal(expected_geometry, mol.geometry.coordinates, decimal=1)

        rmtree("output_files")
        rmtree("error_files")
        rmtree("MD_data")
        rmtree("MD_trajectories")


# Test the simulated_annealing() function on a water molecule in vacuum with no inplace option
def test_DFTBInput_simulated_annealing_no_inplace():

    engine = DFTBInput(parallel="mpi")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        newmol = engine.simulated_annealing(
            mol,
            ncores=1,
            start_temp=1,
            target_temp=2,
            ramp_steps=5,
            hold_steps=10,
            mdrestartfreq=1,
        )
    except:
        assert False, "Unexpected exception raised during simulated annealing"

    else:
        expected_geometry = [
            np.array([-3.20710732, -0.58706447, -0.01012616]),
            np.array([-2.24817069, -0.59594656, -0.02023133]),
            np.array([-3.53500255, -1.23927885, 0.61226748]),
        ]

        # Assertion here is VERY lax due to the nature of the simulation...
        assert_array_almost_equal(expected_geometry, newmol.geometry.coordinates, decimal=1)

        rmtree("output_files")
        rmtree("error_files")
        rmtree("MD_data")
        rmtree("MD_trajectories")



# Test the catching of runtime errors
def test_DFTBInput_runtime_error_input():

    engine = DFTBInput()
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.spe(mol, ncores=4, charge="castoro")
    except:
        assert True
    else:
        assert False, "An exception was not raised on wrong input file."

    for filename in listdir("./"):
        if filename.endswith("_spe"):
            rmtree(filename)
