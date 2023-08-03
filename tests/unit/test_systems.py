import pytest, pathlib, json

from numpy.testing import assert_array_almost_equal
from os.path import abspath, dirname, join

from spycci.config import __JSON_VERSION__
from spycci.systems import System
from spycci.core.base import Engine
from spycci.core.geometry import MolecularGeometry

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))


# Test the System class constructor when loading data from an XYZ file
def test_System_xyz___init__():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")

    expected_coordinates = [
        [-3.21035, -0.58504, -0.01395],
        [-2.24247, -0.61827, 0.01848],
        [-3.4892, -1.24911, 0.63429],
    ]

    try:
        mol = System(xyzfile)

    except:
        assert False, "Exception raised during `System` class constructor"

    assert mol.name == "with_comment"
    assert mol.geometry.atomcount == 3
    assert mol.charge == 0
    assert mol.spin == 1
    assert mol.box_side == None
    assert mol.is_periodic == False

    for i, coord in enumerate(expected_coordinates):
        assert_array_almost_equal(mol.geometry.coordinates[i], coord, decimal=6)


# Test the System class constructor when loading data from a JSON file
def test_System_json___init__():

    jsonfile = join(TEST_DIR, "utils/json_examples/water.json")

    expected_coordinates = [
        [-3.21035, -0.58504, -0.01395],
        [-2.24247, -0.61827, 0.01848],
        [-3.4892, -1.24911, 0.63429],
    ]

    try:
        mol = System(jsonfile)

    except:
        assert False, "Exception raised during `System` class constructor"

    assert mol.name == "test_water"
    assert mol.geometry.atomcount == 3
    assert mol.charge == 0
    assert mol.spin == 1
    assert mol.box_side == None
    assert mol.is_periodic == False

    for i, coord in enumerate(expected_coordinates):
        assert_array_almost_equal(mol.geometry.coordinates[i], coord, decimal=6)


# Test the System class method to save all the system data to a JSON file
def test_System_save_json(tmp_path_factory):

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    folder = tmp_path_factory.mktemp("random_text_files")
    path = join(folder, "water.json")

    mol = System(xyzfile, charge=1, spin=2)
    mol.save_json(path)

    with open(path, "r") as jsonfile:
        data = json.load(jsonfile)

    expected = {
        "__JSON_VERSION__": __JSON_VERSION__, 
        "Box Side": None,
        "Charge": 1,
        "Flags": [],
        "Geometry": {
            "Coordinates": [
                [-3.21035, -0.58504, -0.01395],
                [-2.24247, -0.61827, 0.01848],
                [-3.4892, -1.24911, 0.63429],
            ],
            "Elements list": ["O", "H", "H"],
            "Level of theory geometry": None,
            "Number of atoms": 3,
        },
        "Name": "with_comment",
        "Properties": {
            "Electronic energy (Eh)": None,
            "Gibbs energy (Eh)": None,
            "Helmholtz energy (Eh)": None,
            "Hirshfeld Fukui": {},
            "Hirshfeld charges": [],
            "Hirshfeld spin populations": [],
            "Level of theory electronic": None,
            "Level of theory vibronic": None,
            "Mulliken Fukui": {},
            "Mulliken charges": [],
            "Mulliken spin populations": [],
            "Vibronic energy (Eh)": None,
            "pKa": None,
            "Vibrational data": None,
        },
        "Spin": 2,
    }

    assert data == expected


# Test the geometry property getter and setter of the System class
def test_System_geometry_property():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = System(xyzfile)

    expected_coordinates = [
        [-3.21035, -0.58504, -0.01395],
        [-2.24247, -0.61827, 0.01848],
        [-3.4892, -1.24911, 0.63429],
    ]

    mol.properties.set_electronic_energy(1.5, Engine("Dummy"))

    assert mol.properties.electronic_energy == 1.5
    for i, (_, coord) in enumerate(mol.geometry):
        assert_array_almost_equal(coord, expected_coordinates[i], decimal=6)

    mol.geometry = MolecularGeometry().from_xyz(xyzfile)
    assert mol.properties.electronic_energy == None


def test_System_geometry_property_none_rejection():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = System(xyzfile)

    try:
        mol.geometry = None
    except:
        assert True
    else:
        assert (
            False
        ), "Exception not raised when trying to set the geometry attribute to None"

    try:
        mol.geometry = MolecularGeometry()
    except:
        assert True
    else:
        assert (
            False
        ), "Exception not raised when trying to set the geometry attribute as empty"


# Test the charge property getter and setter of the System class
def test_System_charge_property():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = System(xyzfile)

    mol.properties.set_electronic_energy(1.5, Engine("Dummy"))

    assert mol.charge == 0
    assert mol.properties.electronic_energy == 1.5

    mol.charge = 1
    assert mol.charge == 1
    assert mol.properties.electronic_energy == None


# Test the spin property getter and setter of the System class
def test_System_spin_property():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = System(xyzfile)

    mol.properties.set_electronic_energy(1.5, Engine("Dummy"))

    assert mol.spin == 1
    assert mol.properties.electronic_energy == 1.5

    mol.spin = 2
    assert mol.spin == 2
    assert mol.properties.electronic_energy == None


# Test the box_side property getter and setter of the System class
def test_System_box_side_property():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = System(xyzfile)

    mol.properties.set_electronic_energy(1.5, Engine("Dummy"))

    assert mol.box_side == None
    assert mol.is_periodic == False
    assert mol.properties.electronic_energy == 1.5

    mol.box_side = 10.4
    assert mol.box_side == 10.4
    assert mol.is_periodic == True
    assert mol.properties.electronic_energy == None
