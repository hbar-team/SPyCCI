import pytest, pathlib, json

from numpy.testing import assert_array_almost_equal, assert_almost_equal
from os.path import abspath, dirname, join
from copy import copy, deepcopy

from spycci.config import __JSON_VERSION__
from spycci.systems import System
from spycci.core.base import Engine
from spycci.core.geometry import MolecularGeometry
from spycci.core.properties import Properties

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
    expected = (
        [-0.88340023, -0.17904132, -0.07267199],
        [0.4497649 , 0.51104444, 0.12851809],
        [ 1.48578755, -0.2490953 , -0.47625408],
        [-1.08989689, -0.31470631, -1.13937946],
        [-1.69482305,  0.40270109,  0.37346764],
        [-0.87511854, -1.17633172,  0.37932949],
        [ 0.44081839,  1.50296775, -0.33251669],
        [0.6749398 , 0.62724095, 1.19298181],
        [ 1.49192808, -1.12477959, -0.05347481],
    )

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
            "pKa": {
                "direct": None,
                "oxonium": None,
                "oxonium COSMO-RS": None,
                "free energies": None,
                "level of theory cosmors": None
            },
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
    for i, coord in enumerate(mol.geometry.coordinates):
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


# ----------------------------------------------------------------
#       TEST LISTENER OF THE MOLECULAR GEOMETRY CLASS
# ----------------------------------------------------------------

# Test assignment of the listener of the MolecularGeometry class on __init__
def test_MolecularGeometry_listener___init__():

    geom = MolecularGeometry.from_smiles("C")
    assert geom._MolecularGeometry__system_reset == None

    mol = System("methane", geom)
    assert mol.geometry._MolecularGeometry__system_reset == mol._System__on_geometry_change

# Test assignment of the listener of the MolecularGeometry class on from_xyz
def test_MolecularGeometry_listener_from_xyz():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = System.from_xyz(xyzfile)
    assert mol.geometry._MolecularGeometry__system_reset == mol._System__on_geometry_change


# Test assignment of the listener of the MolecularGeometry class on from_json
def test_MolecularGeometry_listener_from_json():

    jsonfile = join(TEST_DIR, "utils/json_examples/water.json")
    mol = System.from_json(jsonfile)
    assert mol.geometry._MolecularGeometry__system_reset == mol._System__on_geometry_change


# Test assignment of the listener of the MolecularGeometry class on from_smiles
def test_MolecularGeometry_listener_from_smiles():

    mol = System.from_smiles("methane", "C")
    assert mol.geometry._MolecularGeometry__system_reset == mol._System__on_geometry_change


# Test assignment of the listener of the MolecularGeometry class on geometry property assignment
def test_MolecularGeometry_listener_geometry_setter():

    new_geom = MolecularGeometry.from_smiles("c1ccccc1")
    assert new_geom._MolecularGeometry__system_reset == None

    mol = System.from_smiles("methane", "C")
    mol.geometry = new_geom
    assert mol.geometry._MolecularGeometry__system_reset == mol._System__on_geometry_change


# Test the clearing of the listener of the MolecularGeometry class on deepcopy
def test_MolecularGeometry_listener_deepcopy():

    mol = System.from_smiles("methane", "C")
    assert mol.geometry._MolecularGeometry__system_reset == mol._System__on_geometry_change

    geom = mol.geometry
    assert geom._MolecularGeometry__system_reset == mol._System__on_geometry_change

    copy_geom = copy(mol.geometry)
    assert copy_geom._MolecularGeometry__system_reset == mol._System__on_geometry_change

    deepcopy_geom = deepcopy(mol.geometry)
    assert deepcopy_geom._MolecularGeometry__system_reset == None


# Test the __on_geometry_change method of the System class
def test___on_geometry_change_System():

    mol = System.from_smiles("mathane", "C")
    dummy = Engine("dummy engine")
    mol.properties.set_electronic_energy(-1.25, dummy)
    assert mol.adjacency_matrix is not None

    assert_almost_equal(mol.properties.electronic_energy, -1.25, decimal=6)

    mol._System__on_geometry_change()
    assert mol.properties.electronic_energy == None
    assert mol._System__adjacency_matrix is None
    assert mol._System__bond_type_matrix is None


# Test properties clearing on geometry append
def test_System_clearing_geometry_append():

    mol = System.from_smiles("mathane", "C")
    
    # Set one of the properties of the `Property` class
    dummy = Engine("dummy engine")
    mol.properties.set_electronic_energy(-1.25, dummy)
    assert_almost_equal(mol.properties.electronic_energy, -1.25, decimal=6)

    # Change the geometry
    mol.geometry.append("H", [0., 0., 0.])

    # Check that the properties have been cleared
    assert mol.properties.electronic_energy == None


# Test properties clearing on geometry load_xyz
def test_System_clearing_geometry_load_xyz():

    mol = System.from_smiles("methane", "C")
    
    # Set one of the properties of the `Property` class
    dummy = Engine("dummy engine")
    mol.properties.set_electronic_energy(-1.25, dummy)
    assert_almost_equal(mol.properties.electronic_energy, -1.25, decimal=6)

    # Change the geometry
    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol.geometry.load_xyz(xyzfile)

    # Check that the properties have been cleared
    assert mol.properties.electronic_energy == None

# Test property setters
def test_System_clearing_geometry_set_atoms():

    mol = System.from_smiles("methane", "C")

    # Set one of the properties of the `Property` class
    dummy = Engine("dummy engine")
    mol.properties.set_electronic_energy(-1.25, dummy)
    assert_almost_equal(mol.properties.electronic_energy, -1.25, decimal=6)

    # Change atoms list
    mol.geometry.set_atoms(["Sn", "H", "H", "H", "H"])

    assert mol.geometry.atoms == ["Sn", "H", "H", "H", "H"], "Set of the atom list failed"
    assert mol.properties.electronic_energy == None, "Properties not cleared after molecular geometry changed"


# Test property setters
def test_System_clearing_geometry_coordinates_setter():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = System.from_xyz(xyzfile)

    # Set one of the properties of the `Property` class
    dummy = Engine("dummy engine")
    mol.properties.set_electronic_energy(-1.25, dummy)
    assert_almost_equal(mol.properties.electronic_energy, -1.25, decimal=6)

    # Change coordinates using the setter
    new_coordinates = [
        [-10., -0.58504, -0.01395],
        [-2.24247, -0.61827, 0.01848],
        [-3.48920, -1.24911, 0.63429]
    ]

    mol.geometry.set_coordinates(new_coordinates)

    # Check if the coordinates have been set
    assert_array_almost_equal(mol.geometry.coordinates, new_coordinates, decimal=6)
    assert mol.properties.electronic_energy == None, "Properties not cleared after molecular geometry changed"
   

def test_System_connectivity_simple():

    mol = System.from_smiles("formaldehyde", "C=O")

    adj = mol.adjacency_matrix
    bt = mol.bond_type_matrix

    expected_adj = [
        [0., 1., 1., 1.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.]
    ]

    expected_bt = [
        [0., 2., 1., 1.],
        [2., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.]
    ]

    assert_array_almost_equal(adj, expected_adj, decimal=6)
    assert_array_almost_equal(bt, expected_bt, decimal=6) 


def test_System_connectivity_charged():

    geom = MolecularGeometry()
    geom.append("C", [-0.5, 0., 0.])
    geom.append("N", [0.5, 0., 0.])

    mol = System("cyanide", geom, charge=-1)

    adj = mol.adjacency_matrix
    bt = mol.bond_type_matrix

    assert_array_almost_equal(adj, [[0., 1.], [1., 0.]], decimal=6)
    assert_array_almost_equal(bt, [[0., 3.], [3., 0.] ], decimal=6) 


# ----------------------------------------------------------------
#           TEST LISTENER OF THE PROPERTIES CLASS
# ----------------------------------------------------------------

# Test assignment of the listener of the Properties class on __init__
def test_Properties_listener___init__():

    geom = MolecularGeometry.from_smiles("C")
    mol = System("methane", geom)
    assert mol.properties._Properties__check_geometry_level_of_theory == mol._System__check_geometry_level_of_theory


# Test assignment of the listener of the Properties class on from_xyz
def test_Properties_listener_from_xyz():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = System.from_xyz(xyzfile)
    assert mol.properties._Properties__check_geometry_level_of_theory == mol._System__check_geometry_level_of_theory


# Test assignment of the listener of the Properties class on from_json
def test_Properties_listener_from_json():

    jsonfile = join(TEST_DIR, "utils/json_examples/water.json")
    mol = System.from_json(jsonfile)
    assert mol.properties._Properties__check_geometry_level_of_theory == mol._System__check_geometry_level_of_theory


# Test assignment of the listener of the Properties class on from_smiles
def test_Properties_listener_from_smiles():

    mol = System.from_smiles("methane", "C")
    assert mol.properties._Properties__check_geometry_level_of_theory == mol._System__check_geometry_level_of_theory


# Test assignment of the listener of the Properties class on properties assignment
def test_Properties_listener_properties_setter():

    p = Properties()
    assert p._Properties__check_geometry_level_of_theory == None

    mol = System.from_smiles("methane", "C")
    mol.properties = p
    assert mol.properties._Properties__check_geometry_level_of_theory == mol._System__check_geometry_level_of_theory


# Test the clearing of the listener of the Properties class on deepcopy
def test_Properties_listener_deepcopy():

    mol = System.from_smiles("methane", "C")
    assert mol.properties._Properties__check_geometry_level_of_theory == mol._System__check_geometry_level_of_theory

    p = mol.properties
    assert p._Properties__check_geometry_level_of_theory == mol._System__check_geometry_level_of_theory

    copy_p = copy(mol.properties)
    assert copy_p._Properties__check_geometry_level_of_theory == mol._System__check_geometry_level_of_theory

    deepcopy_p = deepcopy(mol.properties)
    assert deepcopy_p._Properties__check_geometry_level_of_theory == None


# Test the __check_geometry_level_of_theory method of the System class in case of level of theory None 
def test___check_geometry_level_of_theory_System_None():

    mol = System.from_smiles("mathane", "C")
    dummy = Engine("Dummy engine")

    try:
        mol._System__check_geometry_level_of_theory(dummy.level_of_theory)
    
    except Exception as e:
        assert False, f"Unexpected exception raised when checking geometry level of theory with None state: {e}"


# Test the __check_geometry_level_of_theory method of the System class in case of same level of theory
def test___check_geometry_level_of_theory_System_match():

    mol = System.from_smiles("mathane", "C")
    dummy = Engine("Dummy engine")
    mol.geometry.level_of_theory_geometry = dummy.level_of_theory

    try:
        mol._System__check_geometry_level_of_theory(dummy.level_of_theory)
    
    except Exception as e:
        assert False, f"Unexpected exception raised when checking matching geometry levels of theory: {e}"


# Test the __check_geometry_level_of_theory method of the System class in case of different level of theory
def test___check_geometry_level_of_theory_System_mismatch():

    mol = System.from_smiles("mathane", "C")

    first = Engine("first engine")
    mol.geometry.level_of_theory_geometry = first.level_of_theory

    second = Engine("second engine")

    try:
        mol._System__check_geometry_level_of_theory(second.level_of_theory)
    
    except:
        assert True
    else:
        assert False, "An exception was not raised when checking mismatching geometry levels of theory."