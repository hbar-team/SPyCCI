import pytest
import shutil
import subprocess

import spycci.config

from numpy.testing import assert_array_almost_equal, assert_almost_equal
from spycci.core.base import Engine
from spycci.core.properties import Properties, pKa

from spycci.engines.xtb import XtbInput


class FakeProc:
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


# Test the Properties class
# ------------------------------------------------------------------------------------------
def test_Properties___init__():

    try:
        Properties()

    except:
        assert False, "Unexpected exception raised on class construction"

    else:
        assert True


def test_Properties_properties():

    p = Properties()

    # Check that all the properties are empty on construction
    assert p.level_of_theory_electronic == None
    assert p.level_of_theory_vibrational == None
    assert p.electronic_energy == None
    assert p.free_energy_correction == None
    assert p.gibbs_free_energy == None
    assert p.mulliken_charges == []
    assert p.mulliken_spin_populations == []
    assert p.condensed_fukui_mulliken == {}
    assert p.hirshfeld_charges == []
    assert p.hirshfeld_spin_populations == []

    # Check that pka attribute is of type pKa and is not set
    assert type(p.pka) == pKa
    assert p.pka.is_set() == False

    # Define a Engine instance to be used in setting the level of theory
    el_engine = Engine("ElMethod")
    vib_engine = Engine("VibMethod")

    # Set all properties
    pka = pKa()
    pka.set_direct(5.0)

    p.set_electronic_energy(1, el_engine)
    p.set_free_energy_correction(0.2, vib_engine)
    p.set_pka(pka, el_engine, vib_engine)
    p.set_mulliken_charges([6, 7, 8], el_engine)
    p.set_mulliken_spin_populations([9, 10, 11], el_engine)
    p.set_condensed_fukui_mulliken({"f+": [0, 1, 2]}, el_engine)
    p.set_hirshfeld_charges([12, 13, 14], el_engine)
    p.set_hirshfeld_spin_populations([15, 16, 17], el_engine)

    # Check that all the properties matces the set values
    assert p.level_of_theory_electronic == el_engine.level_of_theory
    assert p.level_of_theory_vibrational == vib_engine.level_of_theory
    assert p.electronic_energy == 1
    assert p.pka["DIRECT"] == 5.0
    assert_almost_equal(p.free_energy_correction, 0.2, decimal=6)
    assert_almost_equal(p.gibbs_free_energy, 1.2, decimal=6)
    assert_array_almost_equal(p.mulliken_charges, [6, 7, 8], decimal=6)
    assert_array_almost_equal(p.mulliken_spin_populations, [9, 10, 11], decimal=6)
    assert_array_almost_equal(p.condensed_fukui_mulliken["f+"], [0, 1, 2], decimal=6)
    assert_array_almost_equal(p.hirshfeld_charges, [12, 13, 14], decimal=6)
    assert_array_almost_equal(p.hirshfeld_spin_populations, [15, 16, 17], decimal=6)


def test_conflict_electronic():

    p = Properties()
    first = Engine("FirstMethod")
    second = Engine("SecondMethod")

    assert first.level_of_theory != second.level_of_theory

    p.set_electronic_energy(0.1, first)
    p.set_mulliken_charges([1, 2, 3], second)

    assert p.electronic_energy == None
    assert p.level_of_theory_electronic == second.level_of_theory
    assert p.level_of_theory_vibrational == None
    assert_array_almost_equal(p.mulliken_charges, [1, 2, 3], decimal=6)


def test_conflict_vibrational():

    p = Properties()
    first = Engine("FirstMethod")
    second = Engine("SecondMethod")
    pka = pKa()
    pka.set_direct(5.0)

    assert first.level_of_theory != second.level_of_theory

    p.set_free_energy_correction(0.1, first)
    p.set_pka(pka, first, second)

    assert p.free_energy_correction == None
    assert p.level_of_theory_electronic == first.level_of_theory
    assert p.level_of_theory_vibrational == second.level_of_theory
    assert p.pka["DIRECT"] == 5.0


def test_level_of_theory_consistency_normal():

    p = Properties()
    first = Engine("FirstMethod")
    second = Engine("SecondMethod")

    assert spycci.config.STRICTNESS_LEVEL == spycci.config.StrictnessLevel.NORMAL
    assert first.level_of_theory != second.level_of_theory

    p.set_electronic_energy(0.1, first)
    p.set_free_energy_correction(0.02, second)

    assert p.level_of_theory_electronic == first.level_of_theory
    assert p.level_of_theory_vibrational == second.level_of_theory
    assert_almost_equal(p.electronic_energy, 0.1, decimal=6)
    assert_almost_equal(p.free_energy_correction, 0.02, decimal=6)
    assert_almost_equal(p.gibbs_free_energy, 0.12, decimal=6)


def test_level_of_theory_consistency_strict_different_vibrational():

    spycci.config.STRICTNESS_LEVEL = spycci.config.StrictnessLevel.STRICT

    try:
        p = Properties()
        first = Engine("FirstMethod")
        second = Engine("SecondMethod")

        assert spycci.config.STRICTNESS_LEVEL == spycci.config.StrictnessLevel.STRICT
        assert first.level_of_theory != second.level_of_theory

        p.set_electronic_energy(0.1, first)
        p.set_free_energy_correction(0.02, second)

        assert p.level_of_theory_electronic == None
        assert p.level_of_theory_vibrational == second.level_of_theory
        assert p.electronic_energy == None
        assert p.gibbs_free_energy == None
        assert_almost_equal(p.free_energy_correction, 0.02, decimal=6)
    
    finally:
        spycci.config.STRICTNESS_LEVEL = spycci.config.StrictnessLevel.NORMAL


def test_level_of_theory_consistency_strict_different_electronic():

    spycci.config.STRICTNESS_LEVEL = spycci.config.StrictnessLevel.STRICT

    try:
        p = Properties()
        first = Engine("FirstMethod")
        second = Engine("SecondMethod")

        assert spycci.config.STRICTNESS_LEVEL == spycci.config.StrictnessLevel.STRICT
        assert first.level_of_theory != second.level_of_theory

        p.set_free_energy_correction(0.02, second)
        p.set_electronic_energy(0.1, first)

        assert p.level_of_theory_electronic == first.level_of_theory
        assert p.level_of_theory_vibrational == None
        assert p.free_energy_correction == None
        assert p.gibbs_free_energy == None
        assert_almost_equal(p.electronic_energy, 0.1, decimal=6)
    
    finally:
        spycci.config.STRICTNESS_LEVEL = spycci.config.StrictnessLevel.NORMAL


def test_pka_vibrational_addition():

    p = Properties()
    first = Engine("FirstMethod")
    second = Engine("SecondMethod")
    pka = pKa()
    pka.set_direct(4.0)

    assert first.level_of_theory != second.level_of_theory

    p.set_pka(pka, first)

    assert p.pka["DIRECT"] == 4.0
    assert p.level_of_theory_electronic == first.level_of_theory
    assert p.level_of_theory_vibrational == None
    assert p.free_energy_correction == None

    p.set_free_energy_correction(1.0, second)

    assert p.pka.is_set() == False
    assert p.level_of_theory_electronic == first.level_of_theory
    assert p.level_of_theory_vibrational == second.level_of_theory
    assert p.free_energy_correction == 1.0


def test_level_of_theory_consistency_normal_for_mixed_pka():

    p = Properties()
    first = Engine("FirstMethod")
    second = Engine("SecondMethod")

    pka = pKa()
    pka.set_direct(2.5)

    assert spycci.config.STRICTNESS_LEVEL == spycci.config.StrictnessLevel.NORMAL
    assert first.level_of_theory != second.level_of_theory

    p.set_pka(pka, first, second)

    assert p.level_of_theory_electronic == first.level_of_theory
    assert p.level_of_theory_vibrational == second.level_of_theory
    assert_almost_equal(p.pka.direct, 2.5, decimal=6)


def test_level_of_theory_consistency_strict_for_mixed_pka():

    spycci.config.STRICTNESS_LEVEL = spycci.config.StrictnessLevel.STRICT

    try:
        p = Properties()
        first = Engine("FirstMethod")
        second = Engine("SecondMethod")

        pka = pKa()
        pka.set_direct(2.5)

        assert spycci.config.STRICTNESS_LEVEL == spycci.config.StrictnessLevel.NORMAL
        assert first.level_of_theory != second.level_of_theory

        p.set_pka(pka, first, second)
    
    except:
        assert True

    else:
        assert False, "RuntimeEror exception was expected when setting mixed properties with different levels of theory in STRICT mode"
        
    finally:
        spycci.config.STRICTNESS_LEVEL = spycci.config.StrictnessLevel.NORMAL


def test_check_engine():

    p = Properties()

    try:
        p.set_electronic_energy(0.1, "XtbInput || method: gfn2 | solvent: None")
    except:
        assert False, "Exception raised when string is passed to check_engine"

    assert p.level_of_theory_electronic == "XtbInput || method: gfn2 | solvent: None"

    try:
        p.set_electronic_energy(0.1, "This is a string")
    except:
        assert True
    else:
        assert False, "No exception raised when wrong type has been given as engine"

    try:
        p.set_free_energy_correction(0.5, 1)
    except:
        assert True
    else:
        assert False, "No exception raised when wrong type has been given as engine"


# Test the pKa class
# ------------------------------------------------------------------------------------------
def test_pKa___init__():

    try:
        _ = pKa()

    except:
        assert False, "Unexpected exception raised on class construction"

    else:
        assert True


def test_pKa_is_set():

    pka = pKa()
    assert pka.is_set() is False, "Is set function returned True for not set object"

    pka.free_energies = {"SOMETHING": 1.0}
    assert pka.is_set() is True, "Is set function returned False for a set object"


def test_pKa_set_values(monkeypatch):

    # Fake environment
    outputs = {
        "xtb": "* xtb version 6.6.1\n",
    }

    def fake_which(name):
        base = name.split("/")[-1]
        return f"/fake/bin/{base}"

    def fake_run(cmd, capture_output=True, text=True):
        base = cmd[0].split("/")[-1]
        return FakeProc(stdout=outputs.get(base, ""))

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    pka = pKa()
    xtb = XtbInput(XTBPATH="/fake/bin/xtb")

    try:
        pka.set_direct(1.0)

    except:
        assert False, "Exception occured while setting pka direct"

    try:
        pka.set_oxonium(2.0)

    except:
        assert False, "Exception occured while setting pka oxonium"

    try:
        pka.set_oxonium_cormors(3.0, xtb)

    except:
        assert False, "Exception occured while setting pka oxonium COSMO-RS"

    assert_almost_equal(pka.direct, 1.0, decimal=6)
    assert_almost_equal(pka.oxonium, 2.0, decimal=6)
    assert_almost_equal(pka.oxonium_cosmors, 3.0, decimal=6)

    assert_almost_equal(pka["direct"], 1.0, decimal=6)
    assert_almost_equal(pka["oxonium"], 2.0, decimal=6)
    assert_almost_equal(pka["oxonium cosmo-rs"], 3.0, decimal=6)

    assert_almost_equal(pka["DIRECT"], 1.0, decimal=6)
    assert_almost_equal(pka["OXONIUM"], 2.0, decimal=6)
    assert_almost_equal(pka["OXONIUM COSMO-RS"], 3.0, decimal=6)

    assert pka.level_of_theory_cosmors == xtb.level_of_theory
    assert pka.is_set() is True


def test_pKa_wrong_key():

    pka = pKa()

    try:
        _ = pka["WRONG"]
    except:
        assert True
    else:
        assert False, "Exception was not raised when accessing object with wrong key"


def test_pKa___str___set(monkeypatch):

    # Fake environment
    outputs = {
        "xtb": "* xtb version 6.6.1\n",
    }

    def fake_which(name):
        base = name.split("/")[-1]
        return f"/fake/bin/{base}"

    def fake_run(cmd, capture_output=True, text=True):
        base = cmd[0].split("/")[-1]
        return FakeProc(stdout=outputs.get(base, ""))

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    pka = pKa()
    xtb = XtbInput(XTBPATH="/fake/bin/xtb")

    pka.set_direct(1.0)
    pka.set_oxonium(2.0)
    pka.set_oxonium_cormors(3.0, xtb)

    expected_string = ""
    expected_string += "pKa direct: 1.0\n"
    expected_string += "pKa oxonium: 2.0\n"
    expected_string += "pKa oxonium COSMO-RS: 3.0\n"
    expected_string += "COSMO-RS level of theory: XtbInput || method: gfn2 | solvent: None\n"

    assert str(pka) == expected_string, "pKa string representation is different from expected"


def test_pKa___str___not_set():

    pka = pKa()
    expected_string = "pKa object status is NOT SET\n"

    assert str(pka) == expected_string, "pKa string representation is different from expected"


def test_pKa_to_dict(monkeypatch):

    # Fake environment
    outputs = {
        "xtb": "* xtb version 6.6.1\n",
    }

    def fake_which(name):
        base = name.split("/")[-1]
        return f"/fake/bin/{base}"

    def fake_run(cmd, capture_output=True, text=True):
        base = cmd[0].split("/")[-1]
        return FakeProc(stdout=outputs.get(base, ""))

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    pka = pKa()
    xtb = XtbInput(XTBPATH="/fake/bin/xtb")

    pka.set_direct(1.0)
    pka.set_oxonium(2.0)
    pka.set_oxonium_cormors(3.0, xtb)
    pka.free_energies = {"First": 4.0, "Second": 5.0}

    expectd_dict = {
        "direct": 1.0,
        "free energies": {
            "First": 4.0,
            "Second": 5.0,
        },
        "level of theory cosmors": "XtbInput || method: gfn2 | solvent: None",
        "oxonium": 2.0,
        "oxonium COSMO-RS": 3.0,
    }

    out_dict = pka.to_dict()

    for key in out_dict.keys():
        if key not in expectd_dict.keys():
            assert False, "Key mismatch found"

    for key in expectd_dict.keys():
        if key not in out_dict.keys():
            assert False, "Key mismatch found"

    for key, value in expectd_dict.items():
        if key == "free energies":
            for k, v in expectd_dict["free energies"].items():
                assert_almost_equal(v, out_dict["free energies"][k], decimal=6)
        elif key == "level of theory cosmors":
            assert value == out_dict[key], "Mismatch in COSMO-RS level of theory"
        else:
            assert_almost_equal(value, out_dict[key], decimal=6)


def test_pKa_from_dict():

    origin = {
        "direct": 1.0,
        "free energies": {
            "First": 4.0,
            "Second": 5.0,
        },
        "level of theory cosmors": "XtbInput || method: gfn2 | solvent: None",
        "oxonium": 2.0,
        "oxonium COSMO-RS": 3.0,
    }

    try:
        pka = pKa.from_dict(origin)
    except:
        assert False, "Exception raised on pKa initialization using from_dict classmethod"

    assert pka.is_set() is True

    assert_almost_equal(pka.direct, 1.0, decimal=6)
    assert_almost_equal(pka.oxonium, 2.0, decimal=6)
    assert_almost_equal(pka.oxonium_cosmors, 3.0, decimal=6)

    assert pka.level_of_theory_cosmors == origin["level of theory cosmors"], "Mismatch in COSMO-RS level of theory"

    assert_almost_equal(pka.free_energies["First"], 4.0, decimal=6)
    assert_almost_equal(pka.free_energies["Second"], 5.0, decimal=6)
