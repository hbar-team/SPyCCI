import pytest
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

    cube = Cube.from_file(f"{TEST_DIR}/utils/cube_examples/dummy.cube")

    folder = tmp_path_factory.mktemp("tmp_cube_folder")
    path = join(folder, "output.cube")

    cube.save(path, comment_1st="First line", comment_2nd="Second line")

    with open(path, "r") as cubefile:
        data = cubefile.readlines()

    expected = [
        "First line\n",
        "Second line\n",
        "\t2\t0.000000e+00\t0.000000e+00\t0.000000e+00\n",
        "\t3\t1.000000e+00\t0.000000e+00\t0.000000e+00\n",
        "\t3\t0.000000e+00\t1.000000e+00\t0.000000e+00\n",
        "\t3\t0.000000e+00\t0.000000e+00\t1.000000e+00\n",
        "\t1\t-1.000000e-02\t-1.000000e+00\t0.000000e+00\t0.000000e+00\n",
        "\t1\t1.000000e-02\t1.000000e+00\t0.000000e+00\t0.000000e+00\n",
        "\t0.000000e+00\t9.000000e+00\t1.800000e+01\n",
        "\t3.000000e+00\t1.200000e+01\t2.100000e+01\n",
        "\t6.000000e+00\t1.500000e+01\t2.400000e+01\n",
        "\t1.000000e+00\t1.000000e+01\t1.900000e+01\n",
        "\t4.000000e+00\t1.300000e+01\t2.200000e+01\n",
        "\t7.000000e+00\t1.600000e+01\t2.500000e+01\n",
        "\t2.000000e+00\t1.100000e+01\t2.000000e+01\n",
        "\t5.000000e+00\t1.400000e+01\t2.300000e+01\n",
        "\t8.000000e+00\t1.700000e+01\t2.600000e+01\n",
    ]

    assert expected == data
