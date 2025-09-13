import pytest
import numpy as np

from os.path import abspath, dirname, join
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from spycci.tools.cubetools import Cube

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))


# Test the `Cube` class `__init__` mathod
def test_Cube___init__():

    try:
        _ = Cube()

    except:
        assert False, "Exception raised on Cube object construction"


# Test the `from_file` classmethod of the `Cube` class
def test_Cube_from_file():

    try:
        obj = Cube.from_file(f"{TEST_DIR}/utils/cube_examples/dummy.cube")

    except:
        assert False, "Exception raised when using from_file classmethod"

    expected_cube = [
        [[0.0, 9.0, 18.0], [3.0, 12.0, 21.0], [6.0, 15.0, 24.0]],
        [[1.0, 10.0, 19.0], [4.0, 13.0, 22.0], [7.0, 16.0, 25.0]],
        [[2.0, 11.0, 20.0], [5.0, 14.0, 23.0], [8.0, 17.0, 26.0]],
    ]

    assert obj.nvoxels == [3, 3, 3], "Mismatch in the number of voxels"
    assert obj.atomcount == 2
    assert obj.atoms == ["H", "H"], "Mismatch in the atom list"
    assert obj.atomic_numbers == [1, 1], "Mismatch in the atomic number list"

    assert_array_almost_equal(obj.origin, [0.0, 0.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[0], [1.0, 0.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[1], [0.0, 1.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[2], [0.0, 0.0, 1.0], decimal=6)
    assert_array_almost_equal(obj.charges, [-0.01, 0.01], decimal=6)
    assert_array_almost_equal(obj.cube, expected_cube, decimal=6)


# Test the `__getitem__` method of the `Cube` class
def test_Cube___getitem__():

    obj = Cube.from_file(f"{TEST_DIR}/utils/cube_examples/dummy.cube")

    # Test simple access
    assert_almost_equal(obj[1, 0, 0], 1.0, decimal=6)
    assert_almost_equal(obj[0, 1, 0], 3.0, decimal=6)
    assert_almost_equal(obj[0, 0, 1], 9.0, decimal=6)
    assert_almost_equal(obj[2, 0, 1], 11.0, decimal=6)

    # Test 1D slicing
    assert_array_almost_equal(obj[0, 0, :], [0.0, 9.0, 18.0], decimal=6)
    assert_array_almost_equal(obj[1, :, 0], [1.0, 4.0, 7.0], decimal=6)
    assert_array_almost_equal(obj[:, 1, 2], [21.0, 22.0, 23.0], decimal=6)
    assert_array_almost_equal(obj[:, 1, -1], [21.0, 22.0, 23.0], decimal=6)
    assert_array_almost_equal(obj[0, 0, 0:2], [0.0, 9.0], decimal=6)

    # Test 2D slicing
    assert_array_almost_equal(obj[:, :, 0], [[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0]], decimal=6)

    # Test 3D slicing
    assert_array_almost_equal(obj[:, :, :], obj.cube, decimal=6)


# Test the `save` method of the `Cube` class
def test_Cube_save(tmp_path_factory):

    obj = Cube.from_file(f"{TEST_DIR}/utils/cube_examples/dummy.cube")

    folder = tmp_path_factory.mktemp("tmp_cube_folder")
    path = join(folder, "output.cube")

    obj.save(path, comment_1st="First line", comment_2nd="Second line")

    with open(path, "r") as cubefile:
        data = cubefile.readlines()

    expected = [
        "First line\n",
        "Second line\n",
        "2    0.000000e+00    0.000000e+00    0.000000e+00\n",
        "3    1.000000e+00    0.000000e+00    0.000000e+00\n",
        "3    0.000000e+00    1.000000e+00    0.000000e+00\n",
        "3    0.000000e+00    0.000000e+00    1.000000e+00\n",
        "1    -1.000000e-02    -1.000000e+00    0.000000e+00    0.000000e+00\n",
        "1    1.000000e-02    1.000000e+00    0.000000e+00    0.000000e+00\n",
        "0.000000e+00    9.000000e+00    1.800000e+01\n",
        "3.000000e+00    1.200000e+01    2.100000e+01\n",
        "6.000000e+00    1.500000e+01    2.400000e+01\n",
        "1.000000e+00    1.000000e+01    1.900000e+01\n",
        "4.000000e+00    1.300000e+01    2.200000e+01\n",
        "7.000000e+00    1.600000e+01    2.500000e+01\n",
        "2.000000e+00    1.100000e+01    2.000000e+01\n",
        "5.000000e+00    1.400000e+01    2.300000e+01\n",
        "8.000000e+00    1.700000e+01    2.600000e+01\n",
    ]

    assert expected == data


# Test the `scale` method of the `Cube` class
def test_Cube_scale():

    obj = Cube.from_file(f"{TEST_DIR}/utils/cube_examples/dummy.cube")

    try:
        obj = obj.scale(0.25)
    except:
        assert False, "Exception raised when calling scale method"

    expected_cube = np.array([
        [[0.0, 9.0, 18.0], [3.0, 12.0, 21.0], [6.0, 15.0, 24.0]],
        [[1.0, 10.0, 19.0], [4.0, 13.0, 22.0], [7.0, 16.0, 25.0]],
        [[2.0, 11.0, 20.0], [5.0, 14.0, 23.0], [8.0, 17.0, 26.0]],
    ])
    
    expected_cube /= 4.

    assert obj.nvoxels == [3, 3, 3], "Mismatch in the number of voxels"
    assert obj.atomcount == 2
    assert obj.atoms == ["H", "H"], "Mismatch in the atom list"
    assert obj.atomic_numbers == [1, 1], "Mismatch in the atomic number list"

    assert_array_almost_equal(obj.origin, [0.0, 0.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[0], [1.0, 0.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[1], [0.0, 1.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[2], [0.0, 0.0, 1.0], decimal=6)
    assert_array_almost_equal(obj.charges, [-0.01, 0.01], decimal=6)

    assert_array_almost_equal(obj.cube, expected_cube, decimal=6)



# Test the `__add__` method of the `Cube` class
def test_Cube___add__():

    obj_1 = Cube.from_file(f"{TEST_DIR}/utils/cube_examples/dummy.cube")
    obj_2 = obj_1.scale(0.25)

    try:
        obj = obj_1 + obj_2
    except:
        assert False, "Exception raised when adding two cube objects"

    expected_cube = np.array([
        [[0.0, 9.0, 18.0], [3.0, 12.0, 21.0], [6.0, 15.0, 24.0]],
        [[1.0, 10.0, 19.0], [4.0, 13.0, 22.0], [7.0, 16.0, 25.0]],
        [[2.0, 11.0, 20.0], [5.0, 14.0, 23.0], [8.0, 17.0, 26.0]],
    ])
    
    expected_cube *= 1.25

    assert obj.nvoxels == [3, 3, 3], "Mismatch in the number of voxels"
    assert obj.atomcount == 2
    assert obj.atoms == ["H", "H"], "Mismatch in the atom list"
    assert obj.atomic_numbers == [1, 1], "Mismatch in the atomic number list"

    assert_array_almost_equal(obj.origin, [0.0, 0.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[0], [1.0, 0.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[1], [0.0, 1.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[2], [0.0, 0.0, 1.0], decimal=6)
    assert_array_almost_equal(obj.charges, [-0.01, 0.01], decimal=6)

    assert_array_almost_equal(obj.cube, expected_cube, decimal=6)


# Test the `__sub__` method of the `Cube` class
def test_Cube___sub__():

    obj_1 = Cube.from_file(f"{TEST_DIR}/utils/cube_examples/dummy.cube")
    obj_2 = obj_1.scale(0.25)

    try:
        obj = obj_1 - obj_2
    except:
        assert False, "Exception raised when subtracting two cube objects"

    expected_cube = np.array([
        [[0.0, 9.0, 18.0], [3.0, 12.0, 21.0], [6.0, 15.0, 24.0]],
        [[1.0, 10.0, 19.0], [4.0, 13.0, 22.0], [7.0, 16.0, 25.0]],
        [[2.0, 11.0, 20.0], [5.0, 14.0, 23.0], [8.0, 17.0, 26.0]],
    ])
    
    expected_cube *= 0.75

    assert obj.nvoxels == [3, 3, 3], "Mismatch in the number of voxels"
    assert obj.atomcount == 2
    assert obj.atoms == ["H", "H"], "Mismatch in the atom list"
    assert obj.atomic_numbers == [1, 1], "Mismatch in the atomic number list"

    assert_array_almost_equal(obj.origin, [0.0, 0.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[0], [1.0, 0.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[1], [0.0, 1.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[2], [0.0, 0.0, 1.0], decimal=6)
    assert_array_almost_equal(obj.charges, [-0.01, 0.01], decimal=6)

    assert_array_almost_equal(obj.cube, expected_cube, decimal=6)


# Test the `__mul__` method of the `Cube` class
def test_Cube___mul__():

    obj = Cube.from_file(f"{TEST_DIR}/utils/cube_examples/dummy.cube")

    try:
        obj = obj * obj
    except:
        assert False, "Exception raised when multiplying two cube objects"

    expected_cube = np.array([
        [[0.0, 9.0, 18.0], [3.0, 12.0, 21.0], [6.0, 15.0, 24.0]],
        [[1.0, 10.0, 19.0], [4.0, 13.0, 22.0], [7.0, 16.0, 25.0]],
        [[2.0, 11.0, 20.0], [5.0, 14.0, 23.0], [8.0, 17.0, 26.0]],
    ])
    
    expected_cube = expected_cube**2

    assert obj.nvoxels == [3, 3, 3], "Mismatch in the number of voxels"
    assert obj.atomcount == 2
    assert obj.atoms == ["H", "H"], "Mismatch in the atom list"
    assert obj.atomic_numbers == [1, 1], "Mismatch in the atomic number list"

    assert_array_almost_equal(obj.origin, [0.0, 0.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[0], [1.0, 0.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[1], [0.0, 1.0, 0.0], decimal=6)
    assert_array_almost_equal(obj.axes[2], [0.0, 0.0, 1.0], decimal=6)
    assert_array_almost_equal(obj.charges, [-0.01, 0.01], decimal=6)

    assert_array_almost_equal(obj.cube, expected_cube, decimal=6)


