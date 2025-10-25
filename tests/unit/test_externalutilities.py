import pytest, os

from os.path import abspath, dirname, join, isfile

from spycci.systems import System
from spycci.core.base import Engine
from spycci.tools.externalutilities import split_multixyz

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))


# Test the split_multixyz function with default parameters
def test_split_multixyz_default():

    jsonfile = join(TEST_DIR, "utils/json_examples/water.json")
    mol = System.from_json(jsonfile)

    multixyz = join(TEST_DIR, "utils/xyz_examples/multiple.xyz")

    systems = split_multixyz(mol, multixyz)

    assert len(systems) == 4

    obtained_coordinates = [s.geometry.get_coordinates() for s in systems]

    expected_coordinates = [
        [[-3.50000, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
        [[-3.40000, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
        [[-3.30000, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
        [[-3.21035, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
    ]
    
    assert_array_almost_equal(expected_coordinates, obtained_coordinates, decimal=6)

    for n in range(1, 5):
        assert isfile(f"./{mol.name}_{n}.xyz") is True
        os.remove(f"./{mol.name}_{n}.xyz")


# Test the split_multixyz function with suffix
def test_split_multixyz_with_suffix():

    jsonfile = join(TEST_DIR, "utils/json_examples/water.json")
    mol = System.from_json(jsonfile)

    multixyz = join(TEST_DIR, "utils/xyz_examples/multiple.xyz")

    systems = split_multixyz(mol, multixyz, suffix="TEST")

    assert len(systems) == 4

    obtained_coordinates = [s.geometry.get_coordinates() for s in systems]

    expected_coordinates = [
        [[-3.50000, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
        [[-3.40000, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
        [[-3.30000, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
        [[-3.21035, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
    ]
    
    assert_array_almost_equal(expected_coordinates, obtained_coordinates, decimal=6)

    for n in range(1, 5):
        assert isfile(f"./{mol.name}_TEST{n}.xyz") is True
        os.remove(f"./{mol.name}_TEST{n}.xyz")


# Test the split_multixyz function with remove
def test_split_multixyz_with_remove():

    jsonfile = join(TEST_DIR, "utils/json_examples/water.json")
    mol = System.from_json(jsonfile)

    multixyz = join(TEST_DIR, "utils/xyz_examples/multiple.xyz")

    systems = split_multixyz(mol, multixyz, remove_xyz_files=True)

    assert len(systems) == 4

    obtained_coordinates = [s.geometry.get_coordinates() for s in systems]

    expected_coordinates = [
        [[-3.50000, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
        [[-3.40000, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
        [[-3.30000, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
        [[-3.21035, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
    ]
    
    assert_array_almost_equal(expected_coordinates, obtained_coordinates, decimal=6)

    for n in range(1, 5):
        assert isfile(f"./{mol.name}_{n}.xyz") is False


# Test the split_multixyz function with energy parsing from NEB-CI
def test_split_multixyz_with_parsing_NEB_CI():

    jsonfile = join(TEST_DIR, "utils/json_examples/water.json")
    mol = System.from_json(jsonfile)

    dummy_engine = Engine("dummy")

    multixyz = join(TEST_DIR, "utils/xyz_examples/multiple_from_orca_trj.xyz")

    systems = split_multixyz(mol, multixyz, engine=dummy_engine, remove_xyz_files=True)

    assert len(systems) == 4

    obtained_coordinates = [s.geometry.get_coordinates() for s in systems]
    expected_coordinates = [
        [[-3.50000, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
        [[-3.40000, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
        [[-3.30000, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
        [[-3.21035, -0.58504, -0.01395], [-2.24247, -0.61827, 0.01848], [-3.48920, -1.24911, 0.63429]],
    ]
    assert_array_almost_equal(expected_coordinates, obtained_coordinates, decimal=6)
    
    obtained_energies = [s.properties.electronic_energy for s in systems]
    expected_energies = [-111.00, -222.00, -333.00, -444.00]
    assert_array_almost_equal(expected_energies, obtained_energies, decimal=4)