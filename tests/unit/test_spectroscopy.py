import pytest

from os import remove
from os.path import dirname, abspath, isfile

from spycci.systems import System
from spycci.core.spectroscopy import VibrationalData


# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))


def test_VibrationalData___init__():

    try:
        obj = VibrationalData()
    except:
        assert False, "Unexpected exception occurred during contruction"

    assert obj.frequencies == []
    assert obj.normal_modes == []
    assert obj.ir_transitions == []
    assert obj.ir_combination_bands == []
    assert obj.raman_transitions == []


def test_VibrationalData___str__():

    mol = System(f"{TEST_DIR}/utils/json_examples/CO2_ir_raman.json")
    obj = mol.properties.vibrational_data

    expected_string = """VIBRATIONAL FREQUENCIES
----------------------------------------------
 index  frequency  intensity 
         (cm^-1)   (km/mol)  
----------------------------------------------
 0          0.00             
 1          0.00             
 2          0.00             
 3          0.00             
 4          0.00             
 5        623.24      23.18  
 6        626.59      21.94  
 7       1339.10       0.00  
 8       2421.89     489.86  

"""
    assert expected_string == str(obj)


def test_VibrationalData_show_ir_spectrum():

    mol = System(f"{TEST_DIR}/utils/json_examples/CO2_ir_raman.json")
    obj = mol.properties.vibrational_data

    try:
        obj.show_ir_spectrum(
            lineshape="lorentzian",
            include_overtones=True,
            show_bars=True,
            figsize=(8, 4),
            color="red",
            export_path=f"{TEST_DIR}/IR_spectrum_test.png",
            export_dpi=400,
            show=False,
        )

    except:
        assert False, "Unexpected exception occurred during show ir spectrum test"
    
    assert isfile(f"{TEST_DIR}/IR_spectrum_test.png") == True

    remove(f"{TEST_DIR}/IR_spectrum_test.png")


def test_VibrationalData_show_raman_spectrum():

    mol = System(f"{TEST_DIR}/utils/json_examples/CO2_ir_raman.json")
    obj = mol.properties.vibrational_data

    try:
        obj.show_raman_spectrum(
            lineshape="gaussian",
            FWHM=10,
            show_bars=True,
            figsize=(8, 4),
            color="green",
            export_path=f"{TEST_DIR}/Raman_spectrum_test.png",
            export_dpi=400,
            show=False,
        )

    except:
        assert False, "Unexpected exception occurred during show raman spectrum test"
    
    assert isfile(f"{TEST_DIR}/Raman_spectrum_test.png") == True

    remove(f"{TEST_DIR}/Raman_spectrum_test.png")

    
