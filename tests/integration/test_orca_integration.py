import pytest

from spycci.engines.orca import OrcaInput
from spycci.systems import System, ReactionPath
from spycci.tools.externalutilities import split_multixyz
from spycci.core.dependency_finder import locate_orca

from os import listdir
from os.path import dirname, abspath, isfile
from shutil import rmtree
from typing import List

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

# Allow the binding of MPI processes to hardware threads to speed up local testing
import spycci.config
spycci.config.MPI_FLAGS += " --use-hwthread-cpus"

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))

# =================================================================
#       The following tests should be version independent
# =================================================================

# Test the OrcaInput class constructor
# RUNS WITH BOTH ORCA 6.1.0, ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput___init__():
    try:
        engine = OrcaInput(method="HF", basis_set="def2-SVP", aux_basis="def2/J", solvent="water")

    except:
        assert False, "Unenxpected exception raised during OrcaInput class construction"

    else:
        assert engine.method == "HF"
        assert engine.level_of_theory == "OrcaInput || method: HF | basis: def2-SVP | solvent: water"


# Test the spe() function on a radical cation water molecule in DMSO without the inplace option
# RUNS WITH BOTH ORCA 6.1.0, ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_spe_CCSD():
    engine = OrcaInput(method="DLPNO-CCSD", basis_set="def2-SVP", aux_basis="AutoAux")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=1, spin=2)

    try:
        newmol = engine.spe(mol, ncores=8)
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
# RUNS WITH BOTH ORCA 6.1.0, ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_forbidden():
    engine = OrcaInput(method="DLPNO-CCSD(T)", basis_set="6-311++G**", aux_basis="AutoAux")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=1, spin=2)

    try:
        engine.spe(mol, ncores=8, inplace=True)
    except:
        assert False, "Unexpected exception raised"

    assert (
        isfile("./output_files/water_1_2_orca_DLPNO-CCSD-T-_6-311++G--_vacuum_spe.out") == True
    ), "Output file not found"

    rmtree("output_files")


# Test the catching of runtime errors (invalid method)
# RUNS WITH BOTH ORCA 6.1.0, ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_runtime_error_input():
    engine = OrcaInput(method="PBU", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.spe(mol, ncores=8)
    except:
        assert True
    else:
        assert False, "An exception was not raised on wrong input file."

    for filename in listdir("./"):
        if filename.endswith("_spe"):
            rmtree(filename)


# Test the catching of runtime errors (missing basis)
# RUNS WITH BOTH ORCA 6.1.0, ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_runtime_error_missing_basis():
    engine = OrcaInput(method="DLPNO-CCSD", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.spe(mol, ncores=8)
    except:
        assert True
    else:
        assert False, "An exception was not raised on missing basis-set."

    for filename in listdir("./"):
        if filename.endswith("_spe"):
            rmtree(filename)


# Test the catching of runtime errors while testing the block option in the engine init
# RUNS WITH BOTH ORCA 6.1.0, ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_runtime_error_scf_not_converged_in_init():
    engine = OrcaInput(
        method="PBE",
        basis_set="def2-SVP",
        aux_basis="def2/J",
        solvent=None,
        blocks={"scf": {"maxiter": 1}},
    )
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.spe(mol, ncores=8)
    except:
        assert True
    else:
        assert False, "An exception was not raised on SCF not converged."

    for filename in listdir("./"):
        if filename.endswith("_spe"):
            rmtree(filename)


# Test the catching of runtime errors while testing the block option in the engine function call
# RUNS WITH BOTH ORCA 6.1.0, ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_runtime_error_scf_not_converged_in_function():
    engine = OrcaInput(
        method="PBE",
        basis_set="def2-SVP",
        aux_basis="def2/J",
        solvent=None,
    )
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.spe(mol, ncores=8, blocks={"scf": {"maxiter": 1}})
    except:
        assert True
    else:
        assert False, "An exception was not raised on SCF not converged."

    for filename in listdir("./"):
        if filename.endswith("_spe"):
            rmtree(filename)


# Test the catching of runtime errors (wrong multiplicity)
# RUNS WITH BOTH ORCA 6.1.0, ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_runtime_error_wrong_multiplicity():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=0, spin=2)

    try:
        engine.spe(mol, ncores=8)
    except:
        assert True
    else:
        assert False, "An exception was not raised on wrong multiplicity."

    for filename in listdir("./"):
        if filename.endswith("_spe"):
            rmtree(filename)


# =================================================================
#     The following tests have been developed for ORCA 6.1.0
# =================================================================
def check_orca_version():
    try:
        locate_orca(version="==6.1.0")
        return False
    except RuntimeError:
        return True
    
# Test the spe() function on a radical cation water molecule in DMSO
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_spe():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent="DMSO")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=1, spin=2)

    try:
        engine.spe(mol, ncores=8, inplace=True)
    except:
        assert False, "Unexpected exception raised during SPE calculation"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert_array_almost_equal(mol.properties.electronic_energy, -75.942848495681, decimal=6)

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
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_spe_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent="DMSO")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=1, spin=2)

    try:
        newmol = engine.spe(mol, ncores=8)
    except:
        assert False, "Unexpected exception raised during SPE calculation"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory

        assert_array_almost_equal(newmol.properties.electronic_energy, -75.942848495681, decimal=6)

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
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_opt():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.opt(mol, ncores=8, inplace=True)
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
            np.array([-2.244057, -0.623863,  0.023939]),
            np.array([-3.481302, -1.249902,  0.635063]),
        ]
        assert_array_almost_equal(mol.geometry.coordinates, expected_geometry, decimal=6)

        rmtree("output_files")


# Test the opt() function on a water molecule in vacuum with no inplace option
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_opt_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        newmol = engine.opt(mol, ncores=8)
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
            np.array([-2.244057, -0.623863,  0.023939]),
            np.array([-3.481302, -1.249902,  0.635063]),
        ]
        assert_array_almost_equal(expected_geometry, newmol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the opt_ts() function on a water molecule in vacuum
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_opt_ts():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis=None, solvent=None, optionals="D3BJ")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/distorted_TS.xyz", charge=-1, spin=1)

    try:
        engine.opt_ts(mol, ncores=8, inplace=True)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -3073.197345839629, decimal=6)
        assert_almost_equal(mol.properties.free_energy_correction, 0.00555767, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -3073.19178819, decimal=6)

        assert mol.geometry.atoms == ["C", "Br", "H", "H", "H", "Cl"]

        expected_geometry = [
            np.array([-4.346202, 1.272711, -0.022700]),
            np.array([-1.988993, 1.234145, -0.349843]),
            np.array([-4.344809, 2.158810,  0.612816]),
            np.array([-4.399680, 0.286843,  0.440226]),
            np.array([-4.592493, 1.377375, -1.079792]),
            np.array([-6.790102, 1.313545,  0.315858]),
        ]
        
        assert_array_almost_equal(mol.geometry.coordinates, expected_geometry, decimal=6)

        rmtree("output_files")


# Test the opt_ts() function on the distorted TS of the SN2 reaction between bromo methane and the chloride ionin vacuum with no inplace option
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_opt_ts_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis=None, solvent=None, optionals="D3BJ")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/distorted_TS.xyz", charge=-1, spin=1)

    try:
        newmol = engine.opt_ts(mol, ncores=8)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -3073.197345839629, decimal=6)
        assert_almost_equal(newmol.properties.free_energy_correction, 0.00555767, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -3073.19178819, decimal=6)

        assert newmol.geometry.atoms == ["C", "Br", "H", "H", "H", "Cl"]

        expected_geometry = [
            np.array([-4.346202, 1.272711, -0.022700]),
            np.array([-1.988993, 1.234145, -0.349843]),
            np.array([-4.344809, 2.158810,  0.612816]),
            np.array([-4.399680, 0.286843,  0.440226]),
            np.array([-4.592493, 1.377375, -1.079792]),
            np.array([-6.790102, 1.313545,  0.315858]),
        ]

        assert_array_almost_equal(expected_geometry, newmol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the freq() function on a water molecule in vacuum
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_freq():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.freq(mol, ncores=8, inplace=True)
    except:
        assert False, "Unexpected exception raised during frequency analysis"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -76.272562168542, decimal=6)
        assert_almost_equal(mol.properties.free_energy_correction, 0.00328114, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -76.26928151, decimal=6)

        expected_mulliken_charges = np.array([-0.285707, 0.142852, 0.142855])
        assert_array_almost_equal(expected_mulliken_charges, mol.properties.mulliken_charges, decimal=4)

        expected_frequencies = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1571.32, 3754.93, 3869.20]
        assert_array_almost_equal(expected_frequencies, mol.properties.vibrational_data.frequencies, decimal=2)

        expected_modes = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.040730, 0.041229, -0.040246, -0.026996, -0.504366, 0.492341, 0.673468, -0.150019, 0.146437],
            [-0.028761, 0.029314, -0.028615, 0.704419, 0.009214, -0.009000, -0.247918, -0.474481, 0.463171],
            [-0.057255, -0.028855, 0.028168, 0.706300, -0.024224, 0.023641, 0.202449, 0.482213, -0.470718],
        ]

        computed_modes = mol.properties.vibrational_data.normal_modes
        assert len(computed_modes) == 9

        for expected_mode, computed_mode in zip(expected_modes, computed_modes):
            try:
                assert_array_almost_equal(expected_mode, computed_mode, decimal=4)
            except:
                assert_array_almost_equal([-v for v in expected_mode], computed_mode, decimal=4)

        expected_ir_intensities = [(6, 51.23), (7, 2.46), (8, 24.94)]
        computed_ir_intensities = mol.properties.vibrational_data.ir_transitions

        for expected, computed in zip(expected_ir_intensities, computed_ir_intensities):
            assert expected[0] == computed[0]
            assert_almost_equal(expected[1], computed[1], decimal=1)

        rmtree("output_files")


# Test the freq() function on a water molecule in vacuum with no inplace option
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_freq_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        newmol = engine.freq(mol, ncores=8)
    except:
        assert False, "Unexpected exception raised during frequency analysis"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -76.272562168542, decimal=6)
        assert_almost_equal(newmol.properties.free_energy_correction, 0.00328114, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -76.26928151, decimal=6)

        expected_mulliken_charges = np.array([-0.285707, 0.142852, 0.142855])
        assert_array_almost_equal(expected_mulliken_charges, newmol.properties.mulliken_charges, decimal=4)

        rmtree("output_files")


# Test the nfreq() function on a water molecule in ethanol
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_nfreq():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent="ethanol")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.nfreq(mol, ncores=8, inplace=True)
    except:
        assert False, "Unexpected exception raised during numerical frequency analysis"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -76.283375159531, decimal=6)
        assert_almost_equal(mol.properties.free_energy_correction, 0.00321352, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -76.28016164, decimal=6)

        expected_mulliken_charges = np.array([-0.363793, 0.181925, 0.181868])
        assert_array_almost_equal(expected_mulliken_charges, mol.properties.mulliken_charges, decimal=4)

        expected_frequencies = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1535.96, 3766.17, 3863.67]
        assert_array_almost_equal(expected_frequencies, mol.properties.vibrational_data.frequencies, decimal=2)

        expected_modes = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.040429,  0.040978, -0.040001, -0.033429, -0.504185,  0.492165,  0.675121, -0.146219,  0.142728],
            [-0.029004,  0.029717, -0.029008,  0.703193,  0.005115, -0.004999, -0.242847, -0.476779,  0.465414],
            [-0.057335, -0.028773,  0.028087,  0.707601, -0.024538,  0.023948,  0.202427,  0.481224, -0.469753],
        ]

        computed_modes = mol.properties.vibrational_data.normal_modes
        assert len(computed_modes) == 9

        for expected_mode, computed_mode in zip(expected_modes, computed_modes):
            try:
                assert_array_almost_equal(expected_mode, computed_mode, decimal=4)
            except:
                assert_array_almost_equal([-v for v in expected_mode], computed_mode, decimal=4)

        expected_ir_intensities = [(6, 94.71), (7, 21.69), (8, 82.59)]
        computed_ir_intensities = mol.properties.vibrational_data.ir_transitions

        for expected, computed in zip(expected_ir_intensities, computed_ir_intensities):
            assert expected[0] == computed[0]
            assert_almost_equal(expected[1], computed[1], decimal=1)

        rmtree("output_files")


# Test the nfreq() function on a water molecule in ethanol with no inplace option
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_nfreq_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent="ethanol")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        newmol = engine.nfreq(mol, ncores=8)
    except:
        assert False, "Unexpected exception raised during numerical frequency analysis"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibrational == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -76.283375159531, decimal=6)
        assert_almost_equal(newmol.properties.free_energy_correction, 0.00321352, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -76.28016164, decimal=6)

        expected_mulliken_charges = np.array([-0.363793, 0.181925, 0.181868])
        assert_array_almost_equal(expected_mulliken_charges, newmol.properties.mulliken_charges, decimal=4)

        expected_frequencies = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1535.96, 3766.17, 3863.67]
        assert_array_almost_equal(expected_frequencies, newmol.properties.vibrational_data.frequencies, decimal=2)

        expected_modes = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.040429,  0.040978, -0.040001, -0.033429, -0.504185,  0.492165,  0.675121, -0.146219,  0.142728],
            [-0.029004,  0.029717, -0.029008,  0.703193,  0.005115, -0.004999, -0.242847, -0.476779,  0.465414],
            [-0.057335, -0.028773,  0.028087,  0.707601, -0.024538,  0.023948,  0.202427,  0.481224, -0.469753],
        ]

        computed_modes = newmol.properties.vibrational_data.normal_modes
        assert len(computed_modes) == 9

        for expected_mode, computed_mode in zip(expected_modes, computed_modes):
            try:
                assert_array_almost_equal(expected_mode, computed_mode, decimal=4)
            except:
                assert_array_almost_equal([-v for v in expected_mode], computed_mode, decimal=4)

        expected_ir_intensities = [(6, 94.71), (7, 21.69), (8, 82.59)]
        computed_ir_intensities = newmol.properties.vibrational_data.ir_transitions

        for expected, computed in zip(expected_ir_intensities, computed_ir_intensities):
            assert expected[0] == computed[0]
            assert_almost_equal(expected[1], computed[1], decimal=1)

        rmtree("output_files")


# Test the calculation of raman spectra and overtones in orca using a tight optimization
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_raman_nearir():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J")
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/CO2.xyz")

    engine.opt(mol, ncores=8, optimization_level="TIGHTOPT", inplace=True)
    engine.nfreq(mol, ncores=8, raman=True, overtones=True, inplace=True)

    expected_frequencies = [0.00, 0.00, 0.00, 0.00, 0.00, 622.46, 624.16, 1339.45, 2422.18] 
    assert_array_almost_equal(expected_frequencies, mol.properties.vibrational_data.frequencies, decimal=1)

    expected_modes = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.863133, 0.187494, -0.000084, 0.324036, -0.070177, 0.000032, 0.323948, -0.070581, 0.000032],
        [0.000083, -0.000018, -0.883262, -0.000031, 0.000007, 0.331548, -0.000031, 0.000007, 0.331548],
        [-0.000305, 0.000108, 0.000000, -0.150196, -0.690987, -0.000000, 0.150425, 0.690906, -0.000000],
        [0.187494, 0.863133, 0.000000, -0.070371, -0.323967, 0.000000, -0.070386, -0.324017, -0.000000]
    ]

    computed_modes = mol.properties.vibrational_data.normal_modes
    assert len(computed_modes) == 9

    for expected_mode, computed_mode in zip(expected_modes, computed_modes):
        try:
            assert_array_almost_equal(expected_mode, computed_mode, decimal=4)
        except:
            assert_array_almost_equal([-v for v in expected_mode], computed_mode, decimal=4)

    expected_ir_intensities = [(5, 23.17), (6, 23.12), (7, 0.00), (8, 488.46)]
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
        (7, 8, 13.14),
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
# RUNS WITH BOTH ORCA 6.1.0, ORCA 6.0.1 and ORCA 5.0.3
def test_OrcaInput_scan():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        reaction_path: ReactionPath = engine.scan(mol, scan="B 0 1 = 0.8, 1.5, 10", ncores=8)
    except:
        assert False, "Unexpected exception raised during relaxed surface scan"

    else:
        assert len(reaction_path.systems) == 10

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

        calculated_energies = np.array([system.properties.electronic_energy for system in reaction_path.systems])

        assert_array_almost_equal(calculated_energies, expected_energies, decimal=6)

        rmtree("output_files")


# Test the scan_ts() function on a the SN2 reaction between bromo methane and the chloride ion in vacuum
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_scan_ts():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/SN2_scan_example.xyz", charge=-1, spin=1)

    try:
        newmol, reaction_path = engine.scan_ts(mol, scan="B 0 5 = 3.0, 1.0, 10", ncores=8)
    except:
        assert False, "Unexpected exception raised during relaxed surface scan"

    else:
        assert len(reaction_path.systems) == 4

        expected_energies = np.array(
            [
                -3073.194352412784,
                -3073.195142008699,
                -3073.193818936374,
                -3073.195076974921,
            ]
        )

        calculated_energies = np.array([system.properties.electronic_energy for system in reaction_path.systems])

        assert_array_almost_equal(calculated_energies, expected_energies, decimal=6)

        assert_almost_equal(newmol.properties.electronic_energy, -3073.193425502753, decimal=6)
        assert_almost_equal(newmol.properties.free_energy_correction, 0.00552476, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -3073.18790075, decimal=6)

        assert newmol.geometry.atoms == ["C", "Br", "H", "H", "H", "Cl"]

        expected_geometry = [
            np.array([-4.35874488484770, 1.26841036623343,  0.00792242330803]),
            np.array([-1.99961879939520, 1.17581115987002, -0.31131115009977]),
            np.array([-4.34211833467316, 2.17179713069929,  0.61856347241139]),
            np.array([-4.43255182073704, 0.29702533606548,  0.49809991055582]),
            np.array([-4.60057744757888, 1.34816373855040, -1.05246629887291]),
            np.array([-6.80553871276793, 1.36562226858136,  0.33814164269744]),
        ]

        assert_array_almost_equal(expected_geometry, newmol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the scan_ts() function on a the SN2 reaction between bromo methane and the chloride ion in vacuum with inplace option
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_scan_ts_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/SN2_scan_example.xyz", charge=-1, spin=1)

    try:
        engine.scan_ts(mol, scan="B 0 5 = 3.0, 1.0, 10", ncores=8, inplace=True)
    except:
        assert False, "Unexpected exception raised during relaxed surface scan"

    else:
        assert_almost_equal(mol.properties.electronic_energy, -3073.193425502753, decimal=6)
        assert_almost_equal(mol.properties.free_energy_correction, 0.00552476, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -3073.18790075, decimal=6)

        assert mol.geometry.atoms == ["C", "Br", "H", "H", "H", "Cl"]

        expected_geometry = [
            np.array([-4.35874488484770, 1.26841036623343,  0.00792242330803]),
            np.array([-1.99961879939520, 1.17581115987002, -0.31131115009977]),
            np.array([-4.34211833467316, 2.17179713069929,  0.61856347241139]),
            np.array([-4.43255182073704, 0.29702533606548,  0.49809991055582]),
            np.array([-4.60057744757888, 1.34816373855040, -1.05246629887291]),
            np.array([-6.80553871276793, 1.36562226858136,  0.33814164269744]),
        ]

        assert_array_almost_equal(expected_geometry, mol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the OrcaInput IRC function
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_irc():
    engine = OrcaInput(method="PBE", basis_set="def2-SV(P)", aux_basis="def2/J", solvent=None)
    ts = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/IRC_ts.xyz", charge=0, spin=1)

    try:
        MEP_reaction_path: ReactionPath = engine.irc(ts, maxiter=80, ncores=8)
    except:
        assert False, "Exception raised during IRC calculation"

    obtained_systems: List[System] = [s for s in MEP_reaction_path]
    expected_systems: List[System] = split_multixyz(
        ts,
        f"{TEST_DIR}/utils/orca_examples/IRC_MEP_trj.xyz",
        engine=engine,
        remove_xyz_files=True,
    )

    assert len(MEP_reaction_path) == 55

    assert_array_almost_equal(
        [s.properties.electronic_energy for s in obtained_systems],
        [
            -93.234931086174,
            -93.234715689525,
            -93.234085328672,
            -93.233075450423,
            -93.231649855452,
            -93.229916218706,
            -93.227758734533,
            -93.225365490212,
            -93.222552922065,
            -93.219538130938,
            -93.216159648625,
            -93.212609473634,
            -93.208767976946,
            -93.204711358681,
            -93.200488461055,
            -93.19614573822,
            -93.191726700195,
            -93.187283809532,
            -93.182887565714,
            -93.178547546395,
            -93.174430249522,
            -93.170501348335,
            -93.16700046278,
            -93.164366355071,
            -93.159560753036,
            -93.157492617042,
            -93.159255419283,
            -93.162530378344,
            -93.165941462282,
            -93.169793757655,
            -93.173983097401,
            -93.178458363128,
            -93.183207557559,
            -93.187959434411,
            -93.192513587715,
            -93.196696892814,
            -93.200388723194,
            -93.203398652314,
            -93.20551817494,
            -93.20600795614,
            -93.206255562756,
            -93.206711248824,
            -93.207164718983,
            -93.207561747714,
            -93.207901865197,
            -93.208194214661,
            -93.208428793535,
            -93.208609767243,
            -93.208724920311,
            -93.208742123482,
            -93.208748485273,
            -93.208758527026,
            -93.208777198823,
            -93.208793533783,
            -93.208805654133
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)


# Test the OrcaInput NEB function
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_neb():
    engine = OrcaInput(method="PBE", basis_set="def2-SV(P)", aux_basis="def2/J", solvent=None)
    reactant = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_reactant.xyz", charge=0, spin=1)
    product = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_product.xyz", charge=0, spin=1)

    try:
        MEP_reaction_path: ReactionPath = engine.neb('', reactant, product, nimages=5, ncores=8)
    except:
        assert False, "Exception raised during NEB calculation"

    obtained_systems: List[System] = [s for s in MEP_reaction_path]
    expected_systems: List[System] = split_multixyz(
        reactant,
        f"{TEST_DIR}/utils/orca_examples/NEB_MEP_trj.xyz",
        engine=engine,
        remove_xyz_files=True,
    )

    assert len(MEP_reaction_path) == 7

    assert_array_almost_equal(
        [s.properties.electronic_energy for s in obtained_systems],
        [
            -93.234931842405,
            -93.190886806001,
            -93.163342205696,
            -93.158977733645,
            -93.170574202888,
            -93.189129223789,
            -93.208823954532
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)


# Test the OrcaInput NEB-CI function
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_neb_ci():
    engine = OrcaInput(method="PBE", basis_set="def2-SV(P)", aux_basis="def2/J", solvent=None)
    reactant = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_reactant.xyz", charge=0, spin=1)
    product = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_product.xyz", charge=0, spin=1)

    try:
        MEP_reaction_path: ReactionPath = engine.neb('CI', reactant, product, nimages=5, ncores=8)
    except:
        assert False, "Exception raised during NEB-CI calculation"

    obtained_systems: List[System] = [s for s in MEP_reaction_path]
    expected_systems: List[System] = split_multixyz(
        reactant,
        f"{TEST_DIR}/utils/orca_examples/NEB-CI_MEP_trj.xyz",
        engine=engine,
        remove_xyz_files=True,
    )

    assert len(MEP_reaction_path) == 7

    assert_array_almost_equal(
        [s.properties.electronic_energy for s in obtained_systems],
        [
            -93.234931842405,
            -93.193763691212,
            -93.166987209618,
            -93.157492938933,
            -93.167762528503,
            -93.187195751543,
            -93.208823954532
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)


# Test the OrcaInput NEB-TS function
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_neb_ts():
    engine = OrcaInput(method="PBE", basis_set="def2-SV(P)", aux_basis="def2/J", solvent=None)
    reactant = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_reactant.xyz", charge=0, spin=1)
    product = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_product.xyz", charge=0, spin=1)

    try:
        transition_state, MEP_reaction_path = engine.neb('TS', reactant, product, nimages=5, ncores=8)
    except:
        assert False, "Exception raised during NEB-TS calculation"

    obtained_systems: List[System] = [s for s in MEP_reaction_path]
    expected_systems: List[System] = split_multixyz(
        reactant,
        f"{TEST_DIR}/utils/orca_examples/NEB-TS_MEP_trj.xyz",
        engine=engine,
        remove_xyz_files=True,
    )

    assert len(MEP_reaction_path) == 7

    assert_array_almost_equal(
        [s.properties.electronic_energy for s in obtained_systems],
        [
            -93.234931842405,
            -93.19370968548,
            -93.166774914454,
            -93.157488172185,
            -93.168108627705,
            -93.187581273557,
            -93.208823954532
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)

    expected_TS_geometry = [
        [-0.675206,  0.25325 , -0.25325 ],
        [-0.01204 , -0.471865,  0.471865],
        [ 0.687246,  0.218615, -0.218615]
    ]

    assert_array_almost_equal(transition_state.geometry.coordinates, expected_TS_geometry, decimal=6)
    assert_almost_equal(transition_state.properties.electronic_energy, -93.157494167231, decimal=6)

    rmtree("output_files")


# Test the OrcaInput NEB-TS function when providing a transition state guess
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_neb_ts_with_guess():
    engine = OrcaInput(method="PBE", basis_set="def2-SV(P)", aux_basis="def2/J", solvent=None)
    reactant = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_reactant.xyz", charge=0, spin=1)
    product = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_product.xyz", charge=0, spin=1)
    guess = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_ts_guess.xyz", charge=0, spin=1)

    try:
        transition_state, MEP_reaction_path = engine.neb('TS', reactant, product, guess=guess, nimages=5, ncores=8)
    except:
        assert False, "Exception raised during NEB-TS calculation"

    obtained_systems: List[System] = [s for s in MEP_reaction_path]
    expected_systems: List[System] = split_multixyz(
        reactant,
        f"{TEST_DIR}/utils/orca_examples/NEB-TS_with_guess_MEP_trj.xyz",
        engine=engine,
        remove_xyz_files=True,
    )

    assert len(MEP_reaction_path) == 7

    assert_array_almost_equal(
        [s.properties.electronic_energy for s in obtained_systems],
        [
            -93.234931842405,
            -93.193066667152,
            -93.165784079262,
            -93.157572965648,
            -93.168481128046,
            -93.187719020744,
            -93.208823554365
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)

    expected_TS_geometry = [
        [-0.675432,  0.25299 , -0.25299 ],
        [-0.011656, -0.471892,  0.471892],
        [ 0.687088,  0.218902, -0.218902]
    ]

    assert_array_almost_equal(transition_state.geometry.coordinates, expected_TS_geometry, decimal=6)
    assert_almost_equal(transition_state.properties.electronic_energy, -93.157494180856, decimal=6)

    rmtree("output_files")


# Test the OrcaInput ZOOM-NEB-CI function
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_zoom_neb_ci():
    engine = OrcaInput(method="PBE", basis_set="def2-SV(P)", aux_basis="def2/J", solvent=None)
    reactant = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_reactant.xyz", charge=0, spin=1)
    product = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_product.xyz", charge=0, spin=1)

    try:
        MEP_reaction_path: ReactionPath = engine.neb('ZOOM-CI', reactant, product, nimages=5, ncores=8)
    except:
        assert False, "Exception raised during ZOOM-NEB-CI calculation"

    obtained_systems: List[System] = [s for s in MEP_reaction_path]
    expected_systems: List[System] = split_multixyz(
        reactant,
        f"{TEST_DIR}/utils/orca_examples/ZOOM-NEB-CI_MEP_trj.xyz",
        engine=engine,
        remove_xyz_files=True,
    )

    assert len(MEP_reaction_path) == 9

    assert_array_almost_equal(
        [s.properties.electronic_energy for s in obtained_systems],
        [
            -93.234931842405,
            -93.193561143932,
            -93.166914668434,
            -93.159178005995,
            -93.157496674268,
            -93.160285514359,
            -93.167860833205,
            -93.187365956064,
            -93.208823954532
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)


# Test the OrcaInput ZOOM-NEB-TS function
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_zoom_neb_ts():
    engine = OrcaInput(method="PBE", basis_set="def2-SV(P)", aux_basis="def2/J", solvent=None)
    reactant = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_reactant.xyz", charge=0, spin=1)
    product = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_product.xyz", charge=0, spin=1)

    try:
        transition_state, MEP_reaction_path = engine.neb('ZOOM-TS', reactant, product, nimages=5, ncores=8)
    except:
        assert False, "Exception raised during ZOOM-NEB-TS calculation"

    obtained_systems: List[System] = [s for s in MEP_reaction_path]
    expected_systems: List[System] = split_multixyz(
        reactant,
        f"{TEST_DIR}/utils/orca_examples/ZOOM-NEB-TS_MEP_trj.xyz",
        engine=engine,
        remove_xyz_files=True,
    )

    assert len(MEP_reaction_path) == 9

    assert_array_almost_equal(
        [s.properties.electronic_energy for s in obtained_systems],
        [
            -93.234931842405,
            -93.190205440982,
            -93.164472109938,
            -93.158445338898,
            -93.157622796219,
            -93.160646528653,
            -93.168047143114,
            -93.187373358876,
            -93.208823954532
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)

    expected_TS_geometry = [
        [-0.674865,  0.253747, -0.253747],
        [-0.012723, -0.471869,  0.471869],
        [ 0.687588,  0.218122, -0.218122]
    ]

    assert_array_almost_equal(transition_state.geometry.coordinates, expected_TS_geometry, decimal=6)
    assert_almost_equal(transition_state.properties.electronic_energy, -93.157494154183, decimal=6)

    rmtree("output_files")


# Test the OrcaInput NEB-IDPP function
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_OrcaInput_neb_idpp():
    engine = OrcaInput(method="PBE", basis_set="def2-SV(P)", aux_basis="def2/J", solvent=None)
    reactant = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_reactant.xyz", charge=0, spin=1)
    product = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/NEB_product.xyz", charge=0, spin=1)

    try:
        MEP_reaction_path: ReactionPath = engine.neb('IDPP', reactant, product, nimages=5, ncores=8)
    except:
        assert False, "Exception raised during NEB-IDPP calculation"

    obtained_systems: List[System] = [s for s in MEP_reaction_path]
    expected_systems: List[System] = split_multixyz(
        reactant,
        f"{TEST_DIR}/utils/orca_examples/NEB-IDPP_trj.xyz",
        engine=engine,
        remove_xyz_files=True,
    )

    assert len(MEP_reaction_path) == 5

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)


# Test the OrcaInput COSMO-RS function using default settings using built-in water solvent model
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_cosmors_simple():
    
    acetone = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/acetone.xyz", charge=0, spin=1)
    orca = OrcaInput(method="PBE", basis_set="def2-TZVP", solvent="water")

    try:
        G_solvation = orca.cosmors(acetone, solvent="water", ncores=8)

    except:
        assert False, "Unexpected exception raised during COSMO-RS calculation"

    assert_almost_equal(G_solvation, -0.006655741604, decimal=6)

    rmtree("output_files")


# Test the OrcaInput COSMO-RS function using engine level of theory and  built-in water solvent model
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_cosmors_engine_settings():

    acetone = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/acetone.xyz", charge=0, spin=1)
    orca = OrcaInput(method="PBE", basis_set="def2-TZVP", solvent="water")

    try:
        G_solvation = orca.cosmors(acetone, solvent="water", use_engine_settings=True, ncores=8)

    except:
        assert False, "Unexpected exception raised during COSMO-RS calculation"

    assert_almost_equal(G_solvation, -0.005619229758, decimal=6)
    rmtree("output_files")


# Test the OrcaInput COSMO-RS function using default settings using external solvent file
@pytest.mark.skipif(check_orca_version(), reason="Test designed for orca==6.1.0")
def test_cosmors_solventfile():

    acetone = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/acetone.xyz", charge=0, spin=1)
    orca = OrcaInput()

    try:
        solventfile = f"{TEST_DIR}/utils/xyz_files/water.cosmorsxyz"
        G_solvation = orca.cosmors(acetone, solventfile=solventfile, ncores=8)

    except:
        assert False, "Unexpected exception raised during COSMO-RS calculation"

    assert_almost_equal(G_solvation, -0.006666057847, decimal=6)

    rmtree("output_files")