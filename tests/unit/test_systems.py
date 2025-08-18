import pytest, pathlib, json

from numpy.testing import assert_array_almost_equal
from os.path import abspath, dirname, join

from spycci.config import __JSON_VERSION__
from spycci.systems import System
from spycci.core.base import Engine
from spycci.core.geometry import MolecularGeometry

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))


# Test the System class constructor when loading data from a MolecularGeometry object
def test_System_geometry___init__():

    WATER = [
        ["O", -5.02534, 1.26595, 0.01097],
        ["H", -4.05210, 1.22164, -0.01263],
        ["H", -5.30240, 0.44124, -0.42809],
    ]
    
    water = MolecularGeometry()
    for l in WATER:
        water.append(l[0], l[1::])
    
    try:
        mol = System("water", geometry=water)

    except:
        assert False, "Exception raised during `System` class constructor"

    assert mol.name == "water"
    assert mol.geometry.atomcount == 3
    assert mol.charge == 0
    assert mol.spin == 1
    assert mol.box_side == None
    assert mol.is_periodic == False

    for i, coord in enumerate(WATER):
        assert_array_almost_equal(mol.geometry.coordinates[i], coord[1::], decimal=6)


# Test the System class constructor when loading data from an XYZ file
def test_System_from_xyz():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")

    expected_coordinates = [
        [-3.21035, -0.58504, -0.01395],
        [-2.24247, -0.61827, 0.01848],
        [-3.4892, -1.24911, 0.63429],
    ]

    try:
        mol = System.from_xyz(xyzfile)

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
def test_System_from_json():

    jsonfile = join(TEST_DIR, "utils/json_examples/water.json")

    expected_coordinates = [
        [-3.21035, -0.58504, -0.01395],
        [-2.24247, -0.61827, 0.01848],
        [-3.4892, -1.24911, 0.63429],
    ]

    try:
        mol = System.from_json(jsonfile)

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


# Test the System class from_smiles classmethod using a valid SMILES input
def test_System_from_smiles():

    try:
        system = System.from_smiles("ethanol", "CCO", random_seed=1234)
    
    except:
        assert False, "Exception raised during `System` class constructor"

    # Check basic system attributes
    assert system.name == "ethanol"
    assert system.charge == 0
    assert system.spin == 1
    assert system.box_side is None
    assert system.is_periodic is False

    # Check molecular geometry
    geom = system.geometry
    assert geom.atomcount == 9, "Expected 9 atoms for the ethanol molecule"
    assert geom.atoms == ["C", "C", "O", "H", "H", "H", "H", "H", "H"]

    # Check geometry coordinates (with fixed seed)
    expected = [
        [-0.88340023, -0.17904132, -0.07267199],
        [0.4497649 , 0.51104444, 0.12851809],
        [ 1.48578755, -0.2490953 , -0.47625408],
        [-1.08989689, -0.31470631, -1.13937946],
        [-1.69482305,  0.40270109,  0.37346764],
        [-0.87511854, -1.17633172,  0.37932949],
        [ 0.44081839,  1.50296775, -0.33251669],
        [0.6749398 , 0.62724095, 1.19298181],
        [ 1.49192808, -1.12477959, -0.05347481],
    ]

    for i in range(9):
        assert_array_almost_equal(geom.coordinates[i], expected[i], decimal=6)


# Test the System class method to save all the system data to a JSON file
def test_System_save_json(tmp_path_factory):

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    folder = tmp_path_factory.mktemp("random_text_files")
    path = join(folder, "water.json")

    mol = System.from_xyz(xyzfile, charge=1, spin=2)
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
            "Hirshfeld Fukui": {},
            "Hirshfeld charges": [],
            "Hirshfeld spin populations": [],
            "Level of theory electronic": None,
            "Level of theory vibrational": None,
            "Mulliken Fukui": {},
            "Mulliken charges": [],
            "Mulliken spin populations": [],
            "Free energy correction G-E(el) (Eh)": None,
            "pKa": None,
            "Vibrational data": None,
        },
        "Spin": 2,
    }

    assert data == expected


# Test the geometry property getter and setter of the System class
def test_System_geometry_property():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = System.from_xyz(xyzfile)

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
    mol = System.from_xyz(xyzfile)

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
    mol = System.from_xyz(xyzfile)

    mol.properties.set_electronic_energy(1.5, Engine("Dummy"))

    assert mol.charge == 0
    assert mol.properties.electronic_energy == 1.5

    mol.charge = 1
    assert mol.charge == 1
    assert mol.properties.electronic_energy == None


# Test the spin property getter and setter of the System class
def test_System_spin_property():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = System.from_xyz(xyzfile)

    mol.properties.set_electronic_energy(1.5, Engine("Dummy"))

    assert mol.spin == 1
    assert mol.properties.electronic_energy == 1.5

    mol.spin = 2
    assert mol.spin == 2
    assert mol.properties.electronic_energy == None


# Test the box_side property getter and setter of the System class
def test_System_box_side_property():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = System.from_xyz(xyzfile)

    mol.properties.set_electronic_energy(1.5, Engine("Dummy"))

    assert mol.box_side == None
    assert mol.is_periodic == False
    assert mol.properties.electronic_energy == 1.5

    mol.box_side = 10.4
    assert mol.box_side == 10.4
    assert mol.is_periodic == True
    assert mol.properties.electronic_energy == None
