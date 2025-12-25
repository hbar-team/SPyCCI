import pytest
import numpy as np

from numpy.testing import assert_almost_equal, assert_array_almost_equal

from spycci.core.math import unit_vector, distance, angle, dihedral


def test_unit_vector():

    v = np.array([3.0, 0.0, 4.0])
    u = unit_vector(v)

    assert_almost_equal(np.linalg.norm(u), 1.0, decimal=12)
    assert_array_almost_equal(u, [0.6, 0.0, 0.8], decimal=12)

    # Test that normalizing a zero vector raises an exception
    with pytest.raises(ValueError):
        unit_vector(np.array([0.0, 0.0, 0.0]))


def test_distance():
    # Distance along x-axis
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    assert_almost_equal(distance(p1, p2), 1.0, decimal=12)

    # Distance in 3D
    p3 = np.array([1.0, 2.0, 2.0])
    p4 = np.array([4.0, 6.0, 5.0])
    assert_almost_equal(distance(p3, p4), 5.8309518948453, decimal=12)


def test_angle():
    # Right angle (90 degrees)
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([0.0, 1.0, 0.0])
    assert_almost_equal(angle(p1, p2, p3), np.pi / 2, decimal=12)

    # Straight angle (180 degrees)
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([-1.0, 0.0, 0.0])
    assert_almost_equal(angle(p1, p2, p3), np.pi, decimal=12)

    # 60 degrees
    p1 = np.array([0.5, np.sqrt(3) / 2, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([1.0, 0.0, 0.0])
    assert_almost_equal(angle(p1, p2, p3), np.pi / 3, decimal=12)


def test_dihedral():

    # Dihedral 90 degrees
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([1.0, 1.0, 0.0])
    p4 = np.array([1.0, 1.0, 1.0])
    assert_almost_equal(dihedral(p1, p2, p3, p4), np.pi / 2, decimal=12)

    # Dihedral -90 degrees
    p4b = np.array([1.0, 1.0, -1.0])
    assert_almost_equal(dihedral(p1, p2, p3, p4b), -np.pi / 2, decimal=12)

    # Test exception raising with collinear points
    with pytest.raises(RuntimeError):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])
        p4 = np.array([3.0, 0.0, 0.0])
        assert_almost_equal(dihedral(p1, p2, p3, p4), 0.0, decimal=12)
