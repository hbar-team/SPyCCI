import pytest

from spycci.engines.orca import OrcaInput
from spycci.systems import System, Ensemble
from spycci.tools.externalutilities import split_multixyz
from spycci.core.dependency_finder import locate_orca

from os import listdir
from os.path import dirname, abspath, isfile
from shutil import rmtree
from typing import List

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))

# =================================================================
#       The following tests should be version independent
# =================================================================

# Test the OrcaInput class constructor
# RUNS WITH BOTH ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput___init__():
    try:
        engine = OrcaInput(method="HF", basis_set="def2-SVP", aux_basis="def2/J", solvent="water")

    except:
        assert False, "Unenxpected exception raised during OrcaInput class construction"

    else:
        assert engine.method == "HF"
        assert engine.level_of_theory == "OrcaInput || method: HF | basis: def2-SVP | solvent: water"


# Test the spe() function on a radical cation water molecule in DMSO without the inplace option
# RUNS WITH BOTH ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_spe_CCSD():
    engine = OrcaInput(method="DLPNO-CCSD", basis_set="def2-SVP", aux_basis="AutoAux")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=1, spin=2)

    try:
        newmol = engine.spe(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during SPE calculation"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory

        assert_array_almost_equal(newmol.properties.electronic_energy, -75.731114338261, decimal=6)

        expected_mulliken_charges = np.array([0.391458, 0.304269, 0.304274])
        assert_array_almost_equal(
            expected_mulliken_charges,
            newmol.properties.mulliken_charges,
            decimal=4,
        )

        expected_mulliken_spin_populations = np.array([1.061085, -0.030540, -0.030544])
        assert_array_almost_equal(
            expected_mulliken_spin_populations,
            newmol.properties.mulliken_spin_populations,
            decimal=4,
        )

        rmtree("output_files")


# Test that the correct suffix is generated when forbidden symbol is used
# RUNS WITH BOTH ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_forbidden():
    engine = OrcaInput(method="DLPNO-CCSD(T)", basis_set="6-311++G**", aux_basis="AutoAux")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=1, spin=2)

    try:
        engine.spe(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised"

    assert (
        isfile("./output_files/water_1_2_orca_DLPNO-CCSD-T-_6-311++G--_vacuum_spe.out") == True
    ), "Output file not found"

    rmtree("output_files")


# Test the catching of runtime errors (invalid method)
# RUNS WITH BOTH ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_runtime_error_input():
    engine = OrcaInput(method="PBU", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
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


# Test the catching of runtime errors (missing basis)
# RUNS WITH BOTH ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_runtime_error_missing_basis():
    engine = OrcaInput(method="DLPNO-CCSD", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.spe(mol, ncores=4)
    except:
        assert True
    else:
        assert False, "An exception was not raised on missing basis-set."

    for filename in listdir("./"):
        if filename.endswith("_spe"):
            rmtree(filename)


# Test the catching of runtime errors while testing the block option in the engine init
# RUNS WITH BOTH ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_runtime_error_scf_not_converged_in_init():
    engine = OrcaInput(
        method="PBE",
        basis_set="def2-SVP",
        aux_basis="def2/J",
        solvent=None,
        blocks={"scf": {"maxiter": 2}},
    )
    mol = System(f"{TEST_DIR}/utils/xyz_files/europium-aquoion.xyz", charge=3, spin=7)

    try:
        engine.spe(mol, ncores=4)
    except:
        assert True
    else:
        assert False, "An exception was not raised on SCF not converged."

    for filename in listdir("./"):
        if filename.endswith("_spe"):
            rmtree(filename)


# Test the catching of runtime errors while testing the block option in the engine function call
# RUNS WITH BOTH ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_runtime_error_scf_not_converged_in_function():
    engine = OrcaInput(
        method="PBE",
        basis_set="def2-SVP",
        aux_basis="def2/J",
        solvent=None,
    )
    mol = System(f"{TEST_DIR}/utils/xyz_files/europium-aquoion.xyz", charge=3, spin=7)

    try:
        engine.spe(mol, ncores=4, blocks={"scf": {"maxiter": 2}})
    except:
        assert True
    else:
        assert False, "An exception was not raised on SCF not converged."

    for filename in listdir("./"):
        if filename.endswith("_spe"):
            rmtree(filename)


# Test the catching of runtime errors (wrong multiplicity)
# RUNS WITH BOTH ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_runtime_error_wrong_multiplicity():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=0, spin=2)

    try:
        engine.spe(mol, ncores=4)
    except:
        assert True
    else:
        assert False, "An exception was not raised on wrong multiplicity."

    for filename in listdir("./"):
        if filename.endswith("_spe"):
            rmtree(filename)


# =================================================================
#     The following tests have been developed for ORCA 6.0.1
# =================================================================
def check_orca_version():
    try:
        locate_orca(version="==6.0.1")
        return False
    except RuntimeError:
        return True
    
# Test the spe() function on a radical cation water molecule in DMSO
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_spe():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent="DMSO")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=1, spin=2)

    try:
        engine.spe(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during SPE calculation"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert_array_almost_equal(mol.properties.electronic_energy, -75.942846790636, decimal=6)

        expected_mulliken_charges = np.array([0.377367, 0.311323, 0.311309])
        assert_array_almost_equal(expected_mulliken_charges, mol.properties.mulliken_charges, decimal=4)
        expected_mulliken_spin_populations = np.array([1.044689, -0.022344, -0.022345])
        assert_array_almost_equal(
            expected_mulliken_spin_populations,
            mol.properties.mulliken_spin_populations,
            decimal=4,
        )

        rmtree("output_files")


# Test the spe() function on a radical cation water molecule in DMSO without the inplace option
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_spe_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent="DMSO")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=1, spin=2)

    try:
        newmol = engine.spe(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during SPE calculation"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory

        assert_array_almost_equal(newmol.properties.electronic_energy, -75.942846790636, decimal=6)

        expected_mulliken_charges = np.array([0.377367, 0.311323, 0.311309])
        assert_array_almost_equal(
            expected_mulliken_charges,
            newmol.properties.mulliken_charges,
            decimal=4,
        )

        expected_mulliken_spin_populations = np.array([1.044689, -0.022344, -0.022345])
        assert_array_almost_equal(
            expected_mulliken_spin_populations,
            newmol.properties.mulliken_spin_populations,
            decimal=4,
        )

        rmtree("output_files")


# Test the opt() function on a water molecule in vacuum
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_opt():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.opt(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -76.272686996006, decimal=6)
        assert_almost_equal(mol.properties.free_energy_correction, 0.00301258, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -76.26967441, decimal=6)

        expected_mulliken_charges = np.array([-0.285541, 0.142770, 0.142771])
        assert_array_almost_equal(expected_mulliken_charges, mol.properties.mulliken_charges, decimal=4)

        expected_geometry = [
            np.array([-3.216661, -0.578656, -0.020182]),
            np.array([-2.244051, -0.623859, 0.023936]),
            np.array([-3.481308, -1.249905, 0.635066]),
        ]
        assert_array_almost_equal(expected_geometry, mol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the opt() function on a water molecule in vacuum with no inplace option
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_opt_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        newmol = engine.opt(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -76.272686996006, decimal=6)
        assert_almost_equal(newmol.properties.free_energy_correction, 0.00301258, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -76.26967441, decimal=6)

        expected_mulliken_charges = np.array([-0.285541, 0.142770, 0.142771])
        assert_array_almost_equal(expected_mulliken_charges, newmol.properties.mulliken_charges, decimal=4)

        expected_geometry = [
            np.array([-3.216661, -0.578656, -0.020182]),
            np.array([-2.244051, -0.623859, 0.023936]),
            np.array([-3.481308, -1.249905, 0.635066]),
        ]
        assert_array_almost_equal(expected_geometry, newmol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the opt_ts() function on a water molecule in vacuum
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_opt_ts():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis=None, solvent=None, optionals="D3BJ")
    mol = System(f"{TEST_DIR}/utils/xyz_files/distorted_TS.xyz", charge=-1, spin=1)

    try:
        engine.opt_ts(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -3073.197345839629, decimal=6)
        assert_almost_equal(mol.properties.free_energy_correction, 0.00556142, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -3073.19178442, decimal=6)

        assert mol.geometry.atoms == ["C", "Br", "H", "H", "H", "Cl"]

        expected_geometry = [
            np.array([-4.346047, 1.272712, -0.022727]),
            np.array([-1.989319, 1.234146, -0.349789]),
            np.array([-4.344836, 2.158795, 0.612800]),
            np.array([-4.399724, 0.286872, 0.440216]),
            np.array([-4.592513, 1.377378, -1.079771]),
            np.array([-6.789840, 1.313526, 0.315834]),
        ]

        assert_array_almost_equal(expected_geometry, mol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the opt_ts() function on the distorted TS of the SN2 reaction between bromo methane and the chloride ionin vacuum with no inplace option
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_opt_ts_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis=None, solvent=None, optionals="D3BJ")
    mol = System(f"{TEST_DIR}/utils/xyz_files/distorted_TS.xyz", charge=-1, spin=1)

    try:
        newmol = engine.opt_ts(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -3073.197345839629, decimal=6)
        assert_almost_equal(newmol.properties.free_energy_correction, 0.00556142, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -3073.19178442, decimal=6)

        assert newmol.geometry.atoms == ["C", "Br", "H", "H", "H", "Cl"]

        expected_geometry = [
            np.array([-4.346047, 1.272712, -0.022727]),
            np.array([-1.989319, 1.234146, -0.349789]),
            np.array([-4.344836, 2.158795, 0.612800]),
            np.array([-4.399724, 0.286872, 0.440216]),
            np.array([-4.592513, 1.377378, -1.079771]),
            np.array([-6.789840, 1.313526, 0.315834]),
        ]

        assert_array_almost_equal(expected_geometry, newmol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the freq() function on a water molecule in vacuum
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_freq():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.freq(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during frequency analysis"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -76.272562168542, decimal=6)
        assert_almost_equal(mol.properties.free_energy_correction, 0.00327779, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -76.26928438, decimal=6)

        expected_mulliken_charges = np.array([-0.285707, 0.142852, 0.142855])
        assert_array_almost_equal(expected_mulliken_charges, mol.properties.mulliken_charges, decimal=4)

        expected_frequencies = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1571.72, 3753.72, 3868.53]
        assert_array_almost_equal(expected_frequencies, mol.properties.vibrational_data.frequencies, decimal=2)

        expected_modes = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.040746,  0.041229, -0.040246, -0.026744, -0.504373,  0.492348,  0.673469, -0.150011,  0.146429],
            [-0.028658,  0.029355, -0.028655,  0.703230,  0.009338, -0.009121, -0.248367, -0.475262,  0.463933],
            [-0.057297, -0.028812,  0.028125,  0.707529, -0.024113,  0.023533,  0.201892,  0.481415, -0.469939]
        ]

        computed_modes = mol.properties.vibrational_data.normal_modes
        assert len(computed_modes) == 9

        for expected_mode, computed_mode in zip(expected_modes, computed_modes):
            try:
                assert_array_almost_equal(expected_mode, computed_mode, decimal=4)
            except:
                assert_array_almost_equal([-v for v in expected_mode], computed_mode, decimal=4)

        expected_ir_intensities = [(6, 50.98), (7, 2.38), (8, 24.64)]
        computed_ir_intensities = mol.properties.vibrational_data.ir_transitions

        for expected, computed in zip(expected_ir_intensities, computed_ir_intensities):
            assert expected[0] == computed[0]
            assert_almost_equal(expected[1], computed[1], decimal=1)

        rmtree("output_files")


# Test the freq() function on a water molecule in vacuum with no inplace option
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_freq_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        newmol = engine.freq(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during frequency analysis"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -76.272562168542, decimal=6)
        assert_almost_equal(newmol.properties.free_energy_correction, 0.00327779, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -76.26928438, decimal=6)

        expected_mulliken_charges = np.array([-0.285707, 0.142852, 0.142855])
        assert_array_almost_equal(expected_mulliken_charges, newmol.properties.mulliken_charges, decimal=4)

        rmtree("output_files")


# Test the nfreq() function on a water molecule in ethanol
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_nfreq():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent="ethanol")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.nfreq(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during numerical frequency analysis"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -76.283375159531, decimal=6)
        assert_almost_equal(mol.properties.free_energy_correction, 0.00321166, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -76.28016350, decimal=6)

        expected_mulliken_charges = np.array([-0.363793, 0.181925, 0.181868])
        assert_array_almost_equal(expected_mulliken_charges, mol.properties.mulliken_charges, decimal=4)

        expected_frequencies = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1535.51, 3770.02, 3859.45]
        assert_array_almost_equal(expected_frequencies, mol.properties.vibrational_data.frequencies, decimal=2)

        expected_modes = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.040409,  0.041010, -0.040032, -0.033533, -0.504180,  0.492160,  0.674905, -0.146731,  0.143227],
            [-0.030792,  0.028770, -0.028084,  0.725304,  0.004547, -0.004445, -0.236577, -0.461183,  0.450190],
            [-0.056378, -0.029708,  0.029000,  0.684124, -0.025003,  0.024401,  0.210703,  0.496523, -0.484687]
        ]

        computed_modes = mol.properties.vibrational_data.normal_modes
        assert len(computed_modes) == 9

        for expected_mode, computed_mode in zip(expected_modes, computed_modes):
            try:
                assert_array_almost_equal(expected_mode, computed_mode, decimal=4)
            except:
                assert_array_almost_equal([-v for v in expected_mode], computed_mode, decimal=4)

        expected_ir_intensities = [(6, 92.70), (7, 19.70), (8, 78.11)]
        computed_ir_intensities = mol.properties.vibrational_data.ir_transitions

        for expected, computed in zip(expected_ir_intensities, computed_ir_intensities):
            assert expected[0] == computed[0]
            assert_almost_equal(expected[1], computed[1], decimal=1)

        rmtree("output_files")


# Test the nfreq() function on a water molecule in ethanol with no inplace option
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_nfreq_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent="ethanol")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        newmol = engine.nfreq(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during numerical frequency analysis"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -76.283375159531, decimal=6)
        assert_almost_equal(newmol.properties.free_energy_correction, 0.00321166, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -76.28016350, decimal=6)

        expected_mulliken_charges = np.array([-0.363793, 0.181925, 0.181868])
        assert_array_almost_equal(expected_mulliken_charges, newmol.properties.mulliken_charges, decimal=4)

        expected_frequencies = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1535.51, 3770.02, 3859.45]
        assert_array_almost_equal(expected_frequencies, newmol.properties.vibrational_data.frequencies, decimal=2)

        expected_modes = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.040409,  0.041010, -0.040032, -0.033533, -0.504180,  0.492160,  0.674905, -0.146731,  0.143227],
            [-0.030792,  0.028770, -0.028084,  0.725304,  0.004547, -0.004445, -0.236577, -0.461183,  0.450190],
            [-0.056378, -0.029708,  0.029000,  0.684124, -0.025003,  0.024401,  0.210703,  0.496523, -0.484687]
        ]

        computed_modes = newmol.properties.vibrational_data.normal_modes
        assert len(computed_modes) == 9

        for expected_mode, computed_mode in zip(expected_modes, computed_modes):
            try:
                assert_array_almost_equal(expected_mode, computed_mode, decimal=4)
            except:
                assert_array_almost_equal([-v for v in expected_mode], computed_mode, decimal=4)

        expected_ir_intensities = [(6, 92.70), (7, 19.70), (8, 78.11)]
        computed_ir_intensities = newmol.properties.vibrational_data.ir_transitions

        for expected, computed in zip(expected_ir_intensities, computed_ir_intensities):
            assert expected[0] == computed[0]
            assert_almost_equal(expected[1], computed[1], decimal=1)

        rmtree("output_files")


# Test the calculation of raman spectra and overtones in orca using a tight optimization
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_raman_nearir():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J")
    mol = System(f"{TEST_DIR}/utils/xyz_files/CO2.xyz")

    engine.opt(mol, ncores=4, optimization_level="TIGHTOPT", inplace=True)
    engine.nfreq(mol, ncores=4, raman=True, overtones=True, inplace=True)

    expected_frequencies = [0.00, 0.00, 0.00, 0.00, 0.00, 621.03, 622.90, 1349.40, 2421.61] 
    assert_array_almost_equal(expected_frequencies, mol.properties.vibrational_data.frequencies, decimal=1)

    expected_modes = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.863095,  0.187668, -0.000033,  0.323920, -0.070708,  0.000012,  0.324035, -0.070181,  0.000012],
        [0.000032, -0.000007, -0.883262, -0.000012,  0.000003,  0.331548, -0.000012,  0.000003,  0.331548],
        [0.000408, -0.000090, -0.000000, -0.150464, -0.690912,  0.000000,  0.150158,  0.690980,  0.000000],
        [0.187668,  0.863095,  0.000000, -0.070443, -0.323979, -0.000000, -0.070446, -0.323976, -0.000000]
    ]

    computed_modes = mol.properties.vibrational_data.normal_modes
    assert len(computed_modes) == 9

    for expected_mode, computed_mode in zip(expected_modes, computed_modes):
        try:
            assert_array_almost_equal(expected_mode, computed_mode, decimal=4)
        except:
            assert_array_almost_equal([-v for v in expected_mode], computed_mode, decimal=4)

    expected_ir_intensities = [(5, 0.13), (6, 0.14), (7, 0.00), (8, 10.80)]
    computed_ir_intensities = mol.properties.vibrational_data.ir_transitions

    for expected, computed in zip(expected_ir_intensities, computed_ir_intensities):
        assert expected[0] == computed[0]
        assert_almost_equal(expected[1], computed[1], decimal=1)

    expected_ir_overtones = [
        (5, 5, 0.00),
        (5, 6, 0.00),
        (5, 7, 0.04),
        (5, 8, 0.00),
        (6, 6, 0.00),
        (6, 7, 0.04),
        (6, 8, 0.00),
        (7, 7, 0.00),
        (7, 8, 12.95),
        (8, 8, 0.00),
    ]

    computed_ir_overtones = mol.properties.vibrational_data.ir_combination_bands

    for expected, computed in zip(expected_ir_overtones, computed_ir_overtones):
        assert expected[0] == computed[0]
        assert expected[1] == computed[1]
        assert_almost_equal(expected[2], computed[2], decimal=1)
    
    ### ADD TESTS FOR RAMAN PART

    rmtree("output_files")


# Test the scan() function on a water molecule in vacuum
# RUNS WITH BOTH ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_scan():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        ensemble: Ensemble = engine.scan(mol, scan="B 0 1 = 0.8, 1.5, 10", ncores=4)
    except:
        assert False, "Unexpected exception raised during relaxed surface scan"

    else:
        assert len(ensemble.systems) == 10

        expected_energies = np.array(
            [
                -76.23067389,
                -76.26216326,
                -76.27234942,
                -76.27002128,
                -76.26050938,
                -76.24706330,
                -76.23167352,
                -76.21556679,
                -76.19949798,
                -76.18392213,
            ]
        )

        calculated_energies = np.array([system.properties.electronic_energy for system in ensemble.systems])

        assert_array_almost_equal(calculated_energies, expected_energies, decimal=6)

        rmtree("output_files")


# Test the scan_ts() function on a the SN2 reaction between bromo methane and the chloride ion in vacuum
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_scan_ts():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/SN2_scan_example.xyz", charge=-1, spin=1)

    try:
        newmol, ensemble = engine.scan_ts(mol, scan="B 0 5 = 3.0, 1.0, 10", ncores=4)
    except:
        assert False, "Unexpected exception raised during relaxed surface scan"

    else:
        assert len(ensemble.systems) == 4

        expected_energies = np.array(
            [
                -3073.194353552196,
                -3073.195142007526,
                -3073.193818927244,
                -3073.195076976301,
            ]
        )

        calculated_energies = np.array([system.properties.electronic_energy for system in ensemble.systems])

        assert_array_almost_equal(calculated_energies, expected_energies, decimal=6)

        assert_almost_equal(newmol.properties.electronic_energy, -3073.193425489058, decimal=6)
        assert_almost_equal(newmol.properties.free_energy_correction, 0.00551917, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -3073.18790631, decimal=6)

        assert newmol.geometry.atoms == ["C", "Br", "H", "H", "H", "Cl"]

        expected_geometry = [
            np.array([-4.35898160615074, 1.26842582404725,  0.00795577451593]),
            np.array([-1.99911242968795, 1.17583718436398, -0.31136010052162]),
            np.array([-4.34221679588116, 2.17182931971903,  0.61859491486099]),
            np.array([-4.43259919220477, 0.29699769062344,  0.49811111720480]),
            np.array([-4.60065974610042, 1.34816125729991, -1.05248562859407]),
            np.array([-6.80558022997490, 1.36557872394636,  0.33813392253398]),
        ]

        assert_array_almost_equal(expected_geometry, newmol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the scan_ts() function on a the SN2 reaction between bromo methane and the chloride ion in vacuum with inplace option
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_scan_ts_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/SN2_scan_example.xyz", charge=-1, spin=1)

    try:
        engine.scan_ts(mol, scan="B 0 5 = 3.0, 1.0, 10", ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during relaxed surface scan"

    else:
        assert_almost_equal(mol.properties.electronic_energy, -3073.193425489058, decimal=6)
        assert_almost_equal(mol.properties.free_energy_correction, 0.00551917, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -3073.18790631, decimal=6)

        assert mol.geometry.atoms == ["C", "Br", "H", "H", "H", "Cl"]

        expected_geometry = [
            np.array([-4.35898160615074, 1.26842582404725,  0.00795577451593]),
            np.array([-1.99911242968795, 1.17583718436398, -0.31136010052162]),
            np.array([-4.34221679588116, 2.17182931971903,  0.61859491486099]),
            np.array([-4.43259919220477, 0.29699769062344,  0.49811111720480]),
            np.array([-4.60065974610042, 1.34816125729991, -1.05248562859407]),
            np.array([-6.80558022997490, 1.36557872394636,  0.33813392253398]),
        ]

        assert_array_almost_equal(expected_geometry, mol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the OrcaInput NEB-CI function
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_neb_ci():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis=None, solvent=None, optionals="D3BJ")
    reactant = System(f"{TEST_DIR}/utils/xyz_files/NEB_reactant.xyz", charge=0, spin=1)
    product = System(f"{TEST_DIR}/utils/xyz_files/NEB_product.xyz", charge=0, spin=1)

    try:
        MEP_ensemble: Ensemble = engine.neb_ci(reactant, product, nimages=5, ncores=4)
    except:
        assert False, "Exception raised during NEB-CI calculation"

    obtained_systems: List[System] = [s for s in MEP_ensemble]
    expected_systems: List[System] = split_multixyz(
        reactant,
        f"{TEST_DIR}/utils/orca_examples/NEB-CI_MEP_trj.xyz",
        engine=engine,
        remove_xyz_files=True,
    )

    assert len(MEP_ensemble) == 7

    assert_array_almost_equal(
        [s.properties.electronic_energy for s in obtained_systems],
        [
            -153.531198642001,
            -153.499219238244,
            -153.455253437473,
            -153.434521663201,
            -153.459587426533,
            -153.493344941757,
            -153.513745761986,
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)


# Test the OrcaInput NEB-TS function
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_neb_ts():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis=None, solvent=None, optionals="D3BJ")
    reactant = System(f"{TEST_DIR}/utils/xyz_files/NEB_reactant.xyz", charge=0, spin=1)
    product = System(f"{TEST_DIR}/utils/xyz_files/NEB_product.xyz", charge=0, spin=1)

    try:
        transition_state, MEP_ensemble = engine.neb_ts(reactant, product, nimages=5, ncores=4)
    except:
        assert False, "Exception raised during NEB-TS calculation"

    obtained_systems: List[System] = [s for s in MEP_ensemble]
    expected_systems: List[System] = split_multixyz(
        reactant,
        f"{TEST_DIR}/utils/orca_examples/NEB-TS_MEP_trj.xyz",
        engine=engine,
        remove_xyz_files=True,
    )

    assert len(MEP_ensemble) == 7

    assert_array_almost_equal(
        [s.properties.electronic_energy for s in obtained_systems],
        [
            -153.531198642001,
            -153.499982972132,
            -153.455119798438,
            -153.434510993860,
            -153.460338350016,
            -153.492429765640,
            -153.513745761986,
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)

    expected_TS_geometry = [
        [-0.66288375017512,  0.28089789335145, -0.09658592014450],
        [0.50838041291475, -0.51218560403496, -0.16949387099732],
        [0.84030435641920, -0.86024165344363,  0.82930127997872],
        [0.74331161000147,  0.86245495747331,  0.38768486116663],
        [0.69165997418698, -1.24514575519978, -0.97357953744434],
        [-0.45397617395144,  1.36592428687077,  0.56670625350475],
        [-1.66679642939585,  0.10829587498284, -0.54403306606394],
    ]

    assert_array_almost_equal(transition_state.geometry.coordinates, expected_TS_geometry, decimal=6)
    assert_almost_equal(transition_state.properties.electronic_energy, -153.434520429858, decimal=6)

    rmtree("output_files")


# Test the OrcaInput NEB-TS function when providing a transition state guess
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_OrcaInput_neb_ts_with_guess():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis=None, solvent=None, optionals="D3BJ")
    reactant = System(f"{TEST_DIR}/utils/xyz_files/NEB_reactant.xyz", charge=0, spin=1)
    product = System(f"{TEST_DIR}/utils/xyz_files/NEB_product.xyz", charge=0, spin=1)
    guess = System(f"{TEST_DIR}/utils/xyz_files/NEB_ts_guess.xyz", charge=0, spin=1)

    try:
        transition_state, MEP_ensemble = engine.neb_ts(reactant, product, guess=guess, nimages=5, ncores=4)
    except:
        assert False, "Exception raised during NEB-TS calculation"

    obtained_systems: List[System] = [s for s in MEP_ensemble]
    expected_systems: List[System] = split_multixyz(
        reactant,
        f"{TEST_DIR}/utils/orca_examples/NEB-TS_with_guess_MEP_trj.xyz",
        engine=engine,
        remove_xyz_files=True,
    )

    assert len(MEP_ensemble) == 7

    assert_array_almost_equal(
        [s.properties.electronic_energy for s in obtained_systems],
        [
            -153.531198642001,
            -153.498667837873,
            -153.454975008423,
            -153.434532479779,
            -153.461531763554,
            -153.493694365718,
            -153.513745681533,
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)

    expected_TS_geometry = [
        [-0.66307152453482,  0.28004528625208, -0.09713543222412],
        [ 0.50927951618413, -0.51155214737995, -0.16953701064097],
        [ 0.83910731966746, -0.85888550808113,  0.83024738725679],
        [ 0.74224073017020,  0.86284351180684,  0.38832411586614],
        [ 0.69389307522291, -1.24480231392223, -0.97290495232740],
        [-0.45506878741646,  1.36494167662755,  0.56687187045239],
        [-1.66638032929343,  0.10740949469685, -0.54586597838282],

    ]

    assert_array_almost_equal(transition_state.geometry.coordinates, expected_TS_geometry, decimal=6)
    assert_almost_equal(transition_state.properties.electronic_energy, -153.434519670033, decimal=6)

    rmtree("output_files")


# Test the OrcaInput COSMO-RS function using default settings using built-in water solvent model
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_cosmors_simple():
    
    acetone = System(f"{TEST_DIR}/utils/xyz_files/acetone.xyz", charge=0, spin=1)
    orca = OrcaInput(method="PBE", basis_set="def2-TZVP", solvent="water")

    try:
        G_solvation = orca.cosmors(acetone, solvent="water", ncores=4)

    except:
        assert False, "Unexpected exception raised during COSMO-RS calculation"

    assert_almost_equal(G_solvation, -0.006613993346, decimal=6)

    rmtree("output_files")


# Test the OrcaInput COSMO-RS function using engine level of theory and  built-in water solvent model
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_cosmors_engine_settings():

    acetone = System(f"{TEST_DIR}/utils/xyz_files/acetone.xyz", charge=0, spin=1)
    orca = OrcaInput(method="PBE", basis_set="def2-TZVP", solvent="water")

    try:
        G_solvation = orca.cosmors(acetone, solvent="water", use_engine_settings=True, ncores=4)

    except:
        assert False, "Unexpected exception raised during COSMO-RS calculation"

    assert_almost_equal(G_solvation, -0.005607684396, decimal=6)
    rmtree("output_files")


# Test the OrcaInput COSMO-RS function using default settings using external solvent file
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.0.1")
def test_cosmors_solventfile():

    acetone = System(f"{TEST_DIR}/utils/xyz_files/acetone.xyz", charge=0, spin=1)
    orca = OrcaInput()

    try:
        solventfile = f"{TEST_DIR}/utils/xyz_files/water.cosmorsxyz"
        G_solvation = orca.cosmors(acetone, solventfile=solventfile, ncores=4)

    except:
        assert False, "Unexpected exception raised during COSMO-RS calculation"

    assert_almost_equal(G_solvation, -0.006626234385, decimal=6)

    rmtree("output_files")