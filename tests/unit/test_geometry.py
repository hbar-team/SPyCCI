import pytest, pathlib
import numpy as np

from os import listdir
from os.path import abspath, dirname, join
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from spycci.core.geometry import MolecularGeometry

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))

# Test the MolecularGeometry class constructor under normal conditions
def test_MolecularGeometry___init__():
    try:
        MolecularGeometry()
    except:
        assert False, "Exception raised during MolecularGeometry object construction"
    else:
        assert True

# Test the MolecularGeometry class from_xyz classmethod
def test_MolecularGeometry_from_xyz():
    
    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")

    try:
        _ = MolecularGeometry.from_xyz(xyzfile)
    except:
        assert False, "Exception raised during MolecularGeometry object construction"
    else:
        assert True


# Test the MolecularGeometry class from_smiles classmethod
def test_MolecularGeometry_from_smiles():
    smiles = "CCO"  # Ethanol

    try:
        geom = MolecularGeometry.from_smiles(smiles, random_seed=1234)
    except Exception as e:
        assert False, "Exception raised during MolecularGeometry object construction"

    assert geom.atomcount == 9, "9 atoms expected for the ethanol molecule"

    assert geom.atoms == ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    
    expected = [
        np.array([-0.88340023, -0.17904132, -0.07267199]),
        np.array([0.4497649 , 0.51104444, 0.12851809]),
        np.array([ 1.48578755, -0.2490953 , -0.47625408]),
        np.array([-1.08989689, -0.31470631, -1.13937946]),
        np.array([-1.69482305,  0.40270109,  0.37346764]),
        np.array([-0.87511854, -1.17633172,  0.37932949]),
        np.array([ 0.44081839,  1.50296775, -0.33251669]),
        np.array([0.6749398 , 0.62724095, 1.19298181]),
        np.array([ 1.49192808, -1.12477959, -0.05347481])
    ]
    
    assert_array_almost_equal(expected, geom.coordinates, decimal=6)


# Test the MolecularGeometry class from_smiles classmethod with ring torsions
def test_from_smiles_small_ring_torsions_enabled():
    smiles = "C1CC1"  

    try:
        mol = MolecularGeometry.from_smiles(
            smiles,
            use_small_ring_torsions=True,
        )
    except Exception as e:
        assert False, "Exception raised during MolecularGeometry object construction"

    assert mol.atomcount == 9


# Test the MolecularGeometry class from_smiles classmethod with macrocycle torsions
def test_from_smiles_macrocycle_torsions_enabled():
    smiles = "C1CCCCCCCCCCC1"  # anello a 12 atomi

    try:
        mol = MolecularGeometry.from_smiles(
            smiles,
            use_macrocycle_torsions=True,
        )
    except Exception as e:
        assert False, "Exception raised during MolecularGeometry object construction"

    assert mol.atomcount >= 36


# Test the MolecularGeometry load_xyz method
def test_MolecularGeometry_load_xyz():

    folder = join(TEST_DIR, "utils/xyz_examples")
    for xyzfile in listdir(folder):
        print(xyzfile)

        path = join(folder, xyzfile)

        try:
            mol = MolecularGeometry()
            mol.load_xyz(path)
        
        except:
            assert False, f"Exception raised when loading the {xyzfile} file"

        expected = [
            np.array([-3.21035, -0.58504, -0.01395]),
            np.array([-2.24247, -0.61827, 0.01848]),
            np.array([-3.48920, -1.24911, 0.63429])
        ]

        assert mol.atomcount == 3
        assert mol.atoms == ["O", "H", "H"]
        assert_array_almost_equal(expected, mol.coordinates, decimal=6)

# Test the MolecularGeometry class __getitem__, __iter__ and __len__ methods
def test_MolecularGeometry_special_methods():
    
    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = MolecularGeometry.from_xyz(xyzfile)

    # Test the len method
    assert len(mol) == 3

    # Test the getitem method
    atom, coord = mol[1]
    assert atom == "H"
    assert_array_almost_equal(coord, np.array([-2.24247, -0.61827, 0.01848]), decimal=6)

    # Test the failure of the getitem method when calling an invalid index
    try:
        _, _ = mol[3]
    except:
        assert True
    else:
        assert False, "An exception was expected when trying to access index 3"
    
    try:
        _, _ = mol[-1]
    except:
        assert True
    else:
        assert False, "An exception was expected when trying to access index -1"
    
    # Test the iterator method
    expected_atoms = ["O", "H", "H"]
    expected_coordinates = [
        np.array([-3.21035, -0.58504, -0.01395]),
        np.array([-2.24247, -0.61827, 0.01848]),
        np.array([-3.48920, -1.24911, 0.63429])
    ]

    for i, (atom, coord) in enumerate(mol):
        assert expected_atoms[i] == atom
        assert_array_almost_equal(expected_coordinates[i], coord, decimal=6)

# Test the append method
def test_MolecularGeometry_append():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = MolecularGeometry.from_xyz(xyzfile)

    mol.append("Am", [0., 1., 2.])

    assert len(mol) == 4
    assert mol.atomcount == 4
    assert mol.atoms == ["O", "H", "H", "Am"]
    assert mol[3][0] == "Am"
    assert_array_almost_equal(mol[3][1], [0, 1, 2], decimal=6)


# Test the write_xyz method
def test_MolecularGeometry_write_xyz(tmp_path_factory):
    
    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = MolecularGeometry.from_xyz(xyzfile)
    mol.append("N", np.array([0, 0, 0]))

    tmpdir = tmp_path_factory.mktemp("temp_xyz")
    outfile = join(tmpdir, "newxyz.xyz")

    mol.write_xyz(outfile, comment="This is a new xyz file")

    with open(outfile, "r") as file:
        lines = file.readlines()

    assert len(lines) == 6
    assert lines[0] == "4\n"
    assert lines[1] == "This is a new xyz file\n"
    assert lines[2] == "O    -3.2103500000    -0.5850400000    -0.0139500000\n"
    assert lines[3] == "H    -2.2424700000    -0.6182700000    0.0184800000\n"
    assert lines[4] == "H    -3.4892000000    -1.2491100000    0.6342900000\n"
    assert lines[5] == "N    0.0000000000    0.0000000000    0.0000000000\n"


# Test the remaining MolecularGeometry class properties
def test_MolecularGeometry_properties():

    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = MolecularGeometry.from_xyz(xyzfile)

    assert_almost_equal(mol.mass, 18.01528, decimal=4)
    assert mol.atomic_numbers == [8, 1, 1]


def test_MolecularGeometry_center_of_mass():
    
    xyzfile = join(TEST_DIR, "utils/xyz_examples/with_comment.xyz")
    mol = MolecularGeometry.from_xyz(xyzfile)

    expected = [-3.17179934199191, -0.62405335766083, 0.024132922929869]
    assert_almost_equal(mol.center_of_mass, expected, decimal=6)


def test_MolecularGeometry_inertia_tensor_diagonal():

    mol = MolecularGeometry()
    mol.append("H", [-1., 0., 0.])
    mol.append("H", [1., 0., 0.])
    
    assert_almost_equal(mol.center_of_mass, [0., 0., 0.], decimal=6)

    expected_inertia_tensor = [
        [0.00000, 0.00000, 0.00000],
        [0.00000, 2.01588, 0.00000],
        [0.00000, 0.00000, 2.01588]
    ]

    expected_inertia_axes = [
        [1.00000, 0.00000, 0.00000],
        [0.00000, 1.00000, 0.00000],
        [0.00000, 0.00000, 1.00000]
    ]

    assert_array_almost_equal(mol.inertia_tensor, expected_inertia_tensor, decimal=6)
    assert_array_almost_equal(mol.inertia_eigvals, [0.00000, 2.01588, 2.01588], decimal=6)
    assert_array_almost_equal(mol.inertia_eigvecs, expected_inertia_axes, decimal=6)

    expected_rotational_constants = [
        [None, 8.362416993118522, 8.362416993118522],
        [None, 250698.9545187970, 250698.9545187970],
    ]

    for i in range(2):
        for x, y in zip(mol.rotational_constants[i], expected_rotational_constants[i]):
            if x is None:
                assert y is None
            else:
                assert_almost_equal(x, y, decimal=6)
    
    assert mol.rotor_type == "linear rotor", "Wrong type of rotor type found"


def test_MolecularGeometry_inertia_tensor_non_diagonal():

    mol = MolecularGeometry()
    mol.append("H", [-1./np.sqrt(2), -1./np.sqrt(2), 0.])
    mol.append("H", [1./np.sqrt(2), 1./np.sqrt(2), 0.])
    
    assert_almost_equal(mol.center_of_mass, [0., 0., 0.], decimal=6)

    expected_inertia_tensor = [
        [1.00794, -1.00794, 0.00000],
        [-1.00794, 1.00794, 0.00000],
        [0.00000, 0.00000, 2.01588]
    ]

    expected_inertia_axes = [
        [-0.707107, -0.707107, 0.00000],
        [-0.707107, 0.7071070, 0.00000],
        [0.00000, 0.00000, 1.00000]
    ]

    assert_array_almost_equal(mol.inertia_tensor, expected_inertia_tensor, decimal=6)
    assert_array_almost_equal(mol.inertia_eigvals, [0.00000, 2.01588, 2.01588], decimal=6)
    assert_array_almost_equal(mol.inertia_eigvecs, expected_inertia_axes, decimal=6)

    expected_rotational_constants = [
        [None, 8.362416993118522, 8.362416993118522],
        [None, 250698.9545187970, 250698.9545187970],
    ]

    for i in range(2):
        for x, y in zip(mol.rotational_constants[i], expected_rotational_constants[i]):
            if x is None:
                assert y is None
            else:
                assert_almost_equal(x, y, decimal=6)
    
    assert mol.rotor_type == "linear rotor", "Wrong type of rotor type found"


def test_MolecularGeometry_rotor_types():

    mol = MolecularGeometry.from_smiles("C#C")
    assert mol.rotor_type == "linear rotor"

    mol = MolecularGeometry.from_smiles("C")
    assert mol.rotor_type == "spherical top"

    mol = MolecularGeometry.from_smiles("c1ccccc1")
    assert mol.rotor_type == "oblate symmetric top"

    mol = MolecularGeometry.from_smiles("CC#CC")
    assert mol.rotor_type == "prolate symmetric top"

    mol = MolecularGeometry.from_smiles("CC(I)(Br)")
    assert mol.rotor_type == "asymmetric top"


def test_stored_properties_clearing():
    
    mol = MolecularGeometry()
    mol.append("C", [-1., 0., 0.])
    mol.append("N", [1., 0., 0.])

    assert mol.rotor_type == "linear rotor"

    mol.append("H", [0., 1., 0.5])
    assert mol.rotor_type == "asymmetric top"



# Test the MolecularGeometry bureid_volume_fraction method
def test_MolecularGeometry_buried_volume_fraction():

    path = join(TEST_DIR, "utils/xyz_examples/without_comment.xyz")
    
    mol = MolecularGeometry()
    mol.load_xyz(path)
    
    # Test normal buried volume
    bv = mol.buried_volume_fraction(0)
    assert_almost_equal(bv, 0.11210631755, decimal=6)

    # Test buried volume excluding one of the hydrogen atoms
    bv = mol.buried_volume_fraction(0, excluded_atoms=[1])
    assert_almost_equal(bv, 0.06464894707, decimal=6)

    # Test buried volume excluding both the hydrogen atoms
    bv = mol.buried_volume_fraction(0, excluded_atoms=[1, 2])
    assert_almost_equal(bv, 0.0, decimal=6)

    bv = mol.buried_volume_fraction(0, include_hydrogens=False)
    assert_almost_equal(bv, 0.0, decimal=6)



# Test the MolecularGeometry bureid_volume_fraction method with invalid parameters
def test_MolecularGeometry_buried_volume_fraction_fail():

    path = join(TEST_DIR, "utils/xyz_examples/without_comment.xyz")
    
    mol = MolecularGeometry()
    mol.load_xyz(path)

    # Test failure with index out of bounds
    try:
        _ = mol.buried_volume_fraction(4)
    except ValueError:
        assert True
    else:
        assert False, "A `ValueError` exception was expected when accessing non existing atom"
    
    try:
        _ = mol.buried_volume_fraction(-1)
    except ValueError:
        assert True
    else:
        assert False, "A `ValueError` exception was expected when accessing non existing atom"
    
    # Test failure when using unsupported radii type
    try:
        _ = mol.buried_volume_fraction(0, radii_type="wrong")
    except ValueError:
        assert True
    else:
        assert False, "A `ValueError` exception was expected when using wrong radii type"

    # Test failure when using unsupported radii type
    try:
        _ = mol.buried_volume_fraction(0, radii=[1])
    except ValueError:
        assert True
    else:
        assert False, "A `RuntimeError` exception was expected when using wrong radii type"
    
