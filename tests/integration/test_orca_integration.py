import pytest

from spycci.engines.orca import OrcaInput
from spycci.systems import System, Ensemble
from spycci.tools.externalutilities import split_multixyz

from os import listdir
from os.path import dirname, abspath, isfile
from shutil import rmtree
from typing import List

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))


# Test the OrcaInput class constructor
def test_OrcaInput___init__():
    try:
        engine = OrcaInput(method="HF", basis_set="def2-SVP", aux_basis="def2/J", solvent="water")

    except:
        assert False, "Unenxpected exception raised during OrcaInput class construction"

    else:
        assert engine.method == "HF"
        assert engine.level_of_theory == "OrcaInput || method: HF | basis: def2-SVP | solvent: water"


# Test the spe() function on a radical cation water molecule in DMSO
def test_OrcaInput_spe():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent="DMSO")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=1, spin=2)

    try:
        engine.spe(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during SPE calculation"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert_array_almost_equal(mol.properties.electronic_energy, -75.942595825106, decimal=6)

        expected_mulliken_charges = np.array([0.377902, 0.311149, 0.310950])
        assert_array_almost_equal(expected_mulliken_charges, mol.properties.mulliken_charges, decimal=4)
        expected_mulliken_spin_populations = np.array([1.044851, -0.022422, -0.022429])
        assert_array_almost_equal(
            expected_mulliken_spin_populations,
            mol.properties.mulliken_spin_populations,
            decimal=4,
        )

        rmtree("output_files")


# Test the spe() function on a radical cation water molecule in DMSO without the inplace option
def test_OrcaInput_spe_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent="DMSO")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz", charge=1, spin=2)

    try:
        newmol = engine.spe(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during SPE calculation"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory

        assert_array_almost_equal(newmol.properties.electronic_energy, -75.942595825106, decimal=6)

        expected_mulliken_charges = np.array([0.377902, 0.311149, 0.310950])
        assert_array_almost_equal(
            expected_mulliken_charges,
            newmol.properties.mulliken_charges,
            decimal=4,
        )

        expected_mulliken_spin_populations = np.array([1.044851, -0.022422, -0.022429])
        assert_array_almost_equal(
            expected_mulliken_spin_populations,
            newmol.properties.mulliken_spin_populations,
            decimal=4,
        )

        rmtree("output_files")


# Test the spe() function on a radical cation water molecule in DMSO without the inplace option
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


# Test the opt() function on a water molecule in vacuum
def test_OrcaInput_opt():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.opt(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibronic == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -76.272686998306, decimal=6)
        assert_almost_equal(mol.properties.vibronic_energy, 0.00301009, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -76.26967691, decimal=6)

        expected_mulliken_charges = np.array([-0.285546, 0.142772, 0.142773])
        assert_array_almost_equal(expected_mulliken_charges, mol.properties.mulliken_charges, decimal=4)

        expected_geometry = [
            np.array([-3.216653, -0.578663, -0.020175]),
            np.array([-2.244047, -0.623851, 0.023928]),
            np.array([-3.481320, -1.249905, 0.635066]),
        ]
        assert_array_almost_equal(expected_geometry, mol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the opt() function on a water molecule in vacuum with no inplace option
def test_OrcaInput_opt_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        newmol = engine.opt(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibronic == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -76.272686998306, decimal=6)
        assert_almost_equal(newmol.properties.vibronic_energy, 0.00301009, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -76.26967691, decimal=6)

        expected_mulliken_charges = np.array([-0.285546, 0.142772, 0.142773])
        assert_array_almost_equal(expected_mulliken_charges, newmol.properties.mulliken_charges, decimal=4)

        expected_geometry = [
            np.array([-3.216653, -0.578663, -0.020175]),
            np.array([-2.244047, -0.623851, 0.023928]),
            np.array([-3.481320, -1.249905, 0.635066]),
        ]
        assert_array_almost_equal(expected_geometry, newmol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the opt_ts() function on a water molecule in vacuum
def test_OrcaInput_opt_ts():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis=None, solvent=None, optionals="D3BJ")
    mol = System(f"{TEST_DIR}/utils/xyz_files/distorted_TS.xyz", charge=-1, spin=1)

    try:
        engine.opt_ts(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibronic == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -3073.197345858668, decimal=6)
        assert_almost_equal(mol.properties.vibronic_energy, 0.00555863, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -3073.19178723, decimal=6)

        assert mol.geometry.atoms == ["C", "Br", "H", "H", "H", "Cl"]

        expected_geometry = [
            np.array([-4.34617809091678, 1.27270816667045, -0.02270271111085]),
            np.array([-1.98899566021222, 1.23415086905524, -0.34984668381643]),
            np.array([-4.34481452290603, 2.15880376231371, 0.61282000582466]),
            np.array([-4.39967907294042, 0.28683948014245, 0.44022366168219]),
            np.array([-4.59249939135688, 1.37737425228426, -1.07978840801401]),
            np.array([-6.79011123501567, 1.31355200919188, 0.31585904587843]),
        ]

        assert_array_almost_equal(expected_geometry, mol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the opt_ts() function on the distorted TS of the SN2 reaction between bromo methane and the chloride ionin vacuum with no inplace option
def test_OrcaInput_opt_ts_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis=None, solvent=None, optionals="D3BJ")
    mol = System(f"{TEST_DIR}/utils/xyz_files/distorted_TS.xyz", charge=-1, spin=1)

    try:
        newmol = engine.opt_ts(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during geometry optimization"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibronic == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -3073.197345858668, decimal=6)
        assert_almost_equal(newmol.properties.vibronic_energy, 0.00555863, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -3073.19178723, decimal=6)

        assert newmol.geometry.atoms == ["C", "Br", "H", "H", "H", "Cl"]

        expected_geometry = [
            np.array([-4.34617809091678, 1.27270816667045, -0.02270271111085]),
            np.array([-1.98899566021222, 1.23415086905524, -0.34984668381643]),
            np.array([-4.34481452290603, 2.15880376231371, 0.61282000582466]),
            np.array([-4.39967907294042, 0.28683948014245, 0.44022366168219]),
            np.array([-4.59249939135688, 1.37737425228426, -1.07978840801401]),
            np.array([-6.79011123501567, 1.31355200919188, 0.31585904587843]),
        ]

        assert_array_almost_equal(expected_geometry, newmol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the freq() function on a water molecule in vacuum
def test_OrcaInput_freq():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.freq(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during frequency analysis"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibronic == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -76.272562753586, decimal=6)
        assert_almost_equal(mol.properties.vibronic_energy, 0.00327855, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -76.26928420, decimal=6)

        expected_mulliken_charges = np.array([-0.285593, 0.142795, 0.142798])
        assert_array_almost_equal(expected_mulliken_charges, mol.properties.mulliken_charges, decimal=4)

        expected_frequencies = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1571.22, 3754.38, 3868.70]
        assert_array_almost_equal(expected_frequencies, mol.properties.vibrational_data.frequencies, decimal=2)

        expected_modes = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.040743, -0.041219, 0.040236, 0.026863, 0.504371, -0.492346, -0.673538, 0.149853, -0.146275],
            [-0.028793, 0.029302, -0.028604, 0.704773, 0.009164, -0.008952, -0.247768, -0.474253, 0.462948],
            [0.057229, 0.028881, -0.028193, -0.705943, 0.024095, -0.023515, -0.202399, -0.482498, 0.470996],
        ]

        computed_modes = mol.properties.vibrational_data.normal_modes
        assert len(computed_modes) == 9

        for expected_mode, computed_mode in zip(expected_modes, computed_modes):
            try:
                assert_array_almost_equal(expected_mode, computed_mode, decimal=4)
            except:
                assert_array_almost_equal([-v for v in expected_mode], computed_mode, decimal=4)

        expected_ir_intensities = [(6, 51.09), (7, 2.43), (8, 24.85)]
        computed_ir_intensities = mol.properties.vibrational_data.ir_transitions

        for expected, computed in zip(expected_ir_intensities, computed_ir_intensities):
            assert expected[0] == computed[0]
            assert_almost_equal(expected[1], computed[1], decimal=1)

        rmtree("output_files")


# Test the freq() function on a water molecule in vacuum with no inplace option
def test_OrcaInput_freq_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent=None)
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        newmol = engine.freq(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during frequency analysis"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibronic == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -76.272562753586, decimal=6)
        assert_almost_equal(newmol.properties.vibronic_energy, 0.00327855, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -76.26928420, decimal=6)

        expected_mulliken_charges = np.array([-0.285593, 0.142795, 0.142798])
        assert_array_almost_equal(expected_mulliken_charges, newmol.properties.mulliken_charges, decimal=4)

        rmtree("output_files")


# Test the nfreq() function on a water molecule in ethanol
def test_OrcaInput_nfreq():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent="ethanol")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        engine.nfreq(mol, ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during numerical frequency analysis"

    else:
        assert mol.properties.level_of_theory_electronic == engine.level_of_theory
        assert mol.properties.level_of_theory_vibronic == engine.level_of_theory

        assert_almost_equal(mol.properties.electronic_energy, -76.283368184519, decimal=6)
        assert_almost_equal(mol.properties.vibronic_energy, 0.00317889, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -76.28018929, decimal=6)

        expected_mulliken_charges = np.array([-0.363774, 0.181883, 0.181890])
        assert_array_almost_equal(expected_mulliken_charges, mol.properties.mulliken_charges, decimal=4)

        expected_frequencies = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1532.63, 3762.34, 3855.63]
        assert_array_almost_equal(expected_frequencies, mol.properties.vibrational_data.frequencies, decimal=2)

        expected_modes = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.040440, -0.040996, 0.040018, 0.033134, 0.504194, -0.492173, -0.675004, 0.146495, -0.142997],
            [-0.029303, 0.029535, -0.028831, 0.707130, 0.005219, -0.005101, -0.242031, -0.474002, 0.462704],
            [0.057170, 0.028941, -0.028251, -0.703530, 0.024606, -0.024013, -0.203876, -0.483956, 0.472420],
        ]

        computed_modes = mol.properties.vibrational_data.normal_modes
        assert len(computed_modes) == 9

        for expected_mode, computed_mode in zip(expected_modes, computed_modes):
            try:
                assert_array_almost_equal(expected_mode, computed_mode, decimal=4)
            except:
                assert_array_almost_equal([-v for v in expected_mode], computed_mode, decimal=4)

        expected_ir_intensities = [(6, 95.18), (7, 23.64), (8, 93.20)]
        computed_ir_intensities = mol.properties.vibrational_data.ir_transitions

        for expected, computed in zip(expected_ir_intensities, computed_ir_intensities):
            assert expected[0] == computed[0]
            assert_almost_equal(expected[1], computed[1], decimal=1)

        rmtree("output_files")


# Test the nfreq() function on a water molecule in ethanol with no inplace option
def test_OrcaInput_nfreq_no_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J", solvent="ethanol")
    mol = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        newmol = engine.nfreq(mol, ncores=4)
    except:
        assert False, "Unexpected exception raised during numerical frequency analysis"

    else:
        assert newmol.properties.level_of_theory_electronic == engine.level_of_theory
        assert newmol.properties.level_of_theory_vibronic == engine.level_of_theory

        assert_almost_equal(newmol.properties.electronic_energy, -76.283368184519, decimal=6)
        assert_almost_equal(newmol.properties.vibronic_energy, 0.00317889, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -76.28018929, decimal=6)

        expected_mulliken_charges = np.array([-0.363774, 0.181883, 0.181890])
        assert_array_almost_equal(expected_mulliken_charges, newmol.properties.mulliken_charges, decimal=4)

        expected_frequencies = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1532.63, 3762.34, 3855.63]
        assert_array_almost_equal(expected_frequencies, newmol.properties.vibrational_data.frequencies, decimal=2)

        expected_modes = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.040440, -0.040996, 0.040018, 0.033134, 0.504194, -0.492173, -0.675004, 0.146495, -0.142997],
            [-0.029303, 0.029535, -0.028831, 0.707130, 0.005219, -0.005101, -0.242031, -0.474002, 0.462704],
            [0.057170, 0.028941, -0.028251, -0.703530, 0.024606, -0.024013, -0.203876, -0.483956, 0.472420],
        ]

        computed_modes = newmol.properties.vibrational_data.normal_modes
        assert len(computed_modes) == 9

        for expected_mode, computed_mode in zip(expected_modes, computed_modes):
            try:
                assert_array_almost_equal(expected_mode, computed_mode, decimal=4)
            except:
                assert_array_almost_equal([-v for v in expected_mode], computed_mode, decimal=4)

        expected_ir_intensities = [(6, 95.18), (7, 23.64), (8, 93.20)]
        computed_ir_intensities = newmol.properties.vibrational_data.ir_transitions

        for expected, computed in zip(expected_ir_intensities, computed_ir_intensities):
            assert expected[0] == computed[0]
            assert_almost_equal(expected[1], computed[1], decimal=1)

        rmtree("output_files")


# Test the calculation of raman spectra and overtones in orca using a tight optimization
@pytest.mark.xfail
def test_OrcaInput_raman_nearir():
    engine = OrcaInput(method="PBE", basis_set="def2-SVP", aux_basis="def2/J")
    mol = System(f"{TEST_DIR}/utils/xyz_files/CO2.xyz")

    engine.opt(mol, ncores=4, optimization_level="TIGHTOPT", inplace=True)
    engine.nfreq(mol, ncores=4, raman=True, overtones=True, inplace=True)

    expected_frequencies = [0.0, 0.0, 0.0, 0.0, 0.0, 623.22, 624.09, 1339.10, 2421.82]
    assert_array_almost_equal(expected_frequencies, mol.properties.vibrational_data.frequencies, decimal=2)

    expected_modes = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.863117, 0.187569, -0.000001, 0.323987, -0.070401, 0.000001, 0.323984, -0.070413, 0.000001],
        [0.000001, -0.000000, -0.883262, -0.000001, 0.000000, 0.331548, -0.000001, 0.000000, 0.331548],
        [-0.000010, 0.000001, 0.000000, -0.150307, -0.690946, -0.000000, 0.150314, 0.690946, -0.000000],
        [0.187569, 0.863117, 0.000000, -0.070405, -0.323987, -0.000000, -0.070409, -0.323984, -0.000000],
    ]

    computed_modes = mol.properties.vibrational_data.normal_modes
    assert len(computed_modes) == 9

    for expected_mode, computed_mode in zip(expected_modes, computed_modes):
        try:
            assert_array_almost_equal(expected_mode, computed_mode, decimal=4)
        except:
            assert_array_almost_equal([-v for v in expected_mode], computed_mode, decimal=4)

    expected_ir_intensities = [(5, 23.20), (6, 23.20), (7, 0.00), (8, 490.10)]
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
        (7, 8, 13.18),
        (8, 8, 0.00),
    ]

    computed_ir_overtones = mol.properties.vibrational_data.ir_combination_bands

    for expected, computed in zip(expected_ir_overtones, computed_ir_overtones):
        assert expected[0] == computed[0]
        assert expected[1] == computed[1]
        assert_almost_equal(expected[2], computed[2], decimal=1)


# Test the scan() function on a water molecule in vacuum
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
def test_OrcaInput_scan_ts():
    engine = OrcaInput(method="PBE", basis_set="def2-TZVP", aux_basis="def2/J", solvent=None, optionals="D3BJ")
    mol = System(f"{TEST_DIR}/utils/xyz_files/SN2_scan_example.xyz", charge=-1, spin=1)

    try:
        newmol, ensemble = engine.scan_ts(mol, scan="B 0 5 = 3.0, 1.0, 30", ncores=4)
    except:
        assert False, "Unexpected exception raised during relaxed surface scan"

    else:
        assert len(ensemble.systems) == 10

        expected_energies = np.array(
            [
                -3073.73733191,
                -3073.73743862,
                -3073.73738045,
                -3073.73713524,
                -3073.73669352,
                -3073.73606818,
                -3073.73531722,
                -3073.73459907,
                -3073.73419522,
                -3073.73443766,
            ]
        )

        calculated_energies = np.array([system.properties.electronic_energy for system in ensemble.systems])

        assert_array_almost_equal(calculated_energies, expected_energies, decimal=6)

        assert_almost_equal(newmol.properties.electronic_energy, -3073.738116597047, decimal=6)
        assert_almost_equal(newmol.properties.vibronic_energy, 0.00626688, decimal=6)
        assert_almost_equal(newmol.properties.gibbs_free_energy, -3073.73184972, decimal=6)

        assert newmol.geometry.atoms == ["C", "Br", "H", "H", "H", "Cl"]

        expected_geometry = [
            np.array([-4.38574952925426, 1.26951963337438, 0.01156597638921]),
            np.array([-2.01263300836393, 1.17617599984948, -0.30956874117580]),
            np.array([-4.33458395795512, 2.16437759277285, 0.61263889436571]),
            np.array([-4.42432267970508, 0.30449771678974, 0.49316656899693]),
            np.array([-4.59103312695869, 1.34731367857036, -1.04506439211788]),
            np.array([-6.79082769776280, 1.36494537864318, 0.33621169354182]),
        ]

        assert_array_almost_equal(expected_geometry, newmol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the scan_ts() function on a the SN2 reaction between bromo methane and the chloride ion in vacuum with inplace option
def test_OrcaInput_scan_ts_inplace():
    engine = OrcaInput(method="PBE", basis_set="def2-TZVP", aux_basis="def2/J", solvent=None, optionals="D3BJ")
    mol = System(f"{TEST_DIR}/utils/xyz_files/SN2_scan_example.xyz", charge=-1, spin=1)

    try:
        engine.scan_ts(mol, scan="B 0 5 = 3.0, 1.0, 30", ncores=4, inplace=True)
    except:
        assert False, "Unexpected exception raised during relaxed surface scan"

    else:
        assert_almost_equal(mol.properties.electronic_energy, -3073.738116597047, decimal=6)
        assert_almost_equal(mol.properties.vibronic_energy, 0.00626688, decimal=6)
        assert_almost_equal(mol.properties.gibbs_free_energy, -3073.73184972, decimal=6)

        assert mol.geometry.atoms == ["C", "Br", "H", "H", "H", "Cl"]

        expected_geometry = [
            np.array([-4.38574952925426, 1.26951963337438, 0.01156597638921]),
            np.array([-2.01263300836393, 1.17617599984948, -0.30956874117580]),
            np.array([-4.33458395795512, 2.16437759277285, 0.61263889436571]),
            np.array([-4.42432267970508, 0.30449771678974, 0.49316656899693]),
            np.array([-4.59103312695869, 1.34731367857036, -1.04506439211788]),
            np.array([-6.79082769776280, 1.36494537864318, 0.33621169354182]),
        ]

        assert_array_almost_equal(expected_geometry, mol.geometry.coordinates, decimal=6)

        rmtree("output_files")


# Test the catching of runtime errors (invalid method)
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


# Test the OrcaInput NEB-CI function
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
            -153.531198654272,
            -153.499682766649,
            -153.455008967825,
            -153.434520027972,
            -153.460090234535,
            -153.492684796405,
            -153.513745795335,
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)


# Test the OrcaInput NEB-TS function
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
            -153.531198654272,
            -153.500678708877,
            -153.457297578306,
            -153.434519231762,
            -153.458457537182,
            -153.493079403986,
            -153.513745795335,
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)

    expected_TS_geometry = [
        [-0.66292173004488, 0.28020099391560, -0.09688510457409],
        [0.50864138286520, -0.51262321034838, -0.16942824046659],
        [0.84004819509531, -0.85971567980890, 0.82996893940146],
        [0.74460764928771, 0.86471255892441, 0.38867735697469],
        [0.69231701008033, -1.24596948472798, -0.97309427425665],
        [-0.45629726852869, 1.36572468570753, 0.56620484824029],
        [-1.66639523875499, 0.10767013633772, -0.54544352531911],
    ]

    assert_array_almost_equal(transition_state.geometry.coordinates, expected_TS_geometry, decimal=6)
    assert_almost_equal(transition_state.properties.electronic_energy, -153.434523068242, decimal=6)

    rmtree("output_files")


# Test the OrcaInput NEB-TS function when providing a transition state guess
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
            -153.531198654272,
            -153.497043001606,
            -153.452461191009,
            -153.434512474274,
            -153.462105880760,
            -153.493181129378,
            -153.513745715132,
        ],
        decimal=6,
    )

    for obtained, expected in zip(obtained_systems, expected_systems):
        assert obtained.geometry.atomcount == expected.geometry.atomcount
        assert_array_almost_equal(obtained.geometry.coordinates, expected.geometry.coordinates, decimal=6)

    expected_TS_geometry = [
        [-0.66258722569397, 0.28074415995905, -0.09803929449706],
        [0.50878195281299, -0.51260563155265, -0.16941970531052],
        [0.83808510670708, -0.85951988550587, 0.83075709824131],
        [0.74423907203868, 0.86444249044971, 0.38945188613164],
        [0.69340150942614, -1.24597256439708, -0.97273136984476],
        [-0.45605583581555, 1.36528811387283, 0.56671312004464],
        [-1.66586457947537, 0.10762331717400, -0.54673173476526],
    ]

    assert_array_almost_equal(transition_state.geometry.coordinates, expected_TS_geometry, decimal=6)
    assert_almost_equal(transition_state.properties.electronic_energy, -153.434522931481, decimal=6)

    rmtree("output_files")
