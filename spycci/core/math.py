import numpy as np


def unit_vector(v: np.ndarray) -> np.ndarray:
    """
    Given a vector `v` computes the unit vector as `v/np.linalg.norm(v)`

    Arguments
    ---------
    v: np.ndarray
        The vector for which the unit vector must be computed
    
    Returns
    -------
    np.ndarray
        The unit vector along the direction of `v`
    """
    norm = np.linalg.norm(v)

    if norm == 0.:
        raise ValueError("Cannot compute the norm of a vector with norm zero.")

    return v / norm


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Computes the euclidian distance between two points (`p1` and `p2`) in the 3D space.

    Arguments
    ---------
    p1: np.ndarray
        The numpy array of shape (3,) encoding the first point in space.
    p2: np.ndarray
        The numpy array of shape (3,) encoding the second point in space.
    
    Returns
    -------
    float
        The distance between the two points in space.
    """
    return np.linalg.norm(p2-p1)


def angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Computes the angle (in radiants) formed by the points (`p1`, `p2` and `p3`) where
    `p2` represents the vertex point.

    Arguments
    ---------
    p1: np.ndarray
        The numpy array of shape (3,) encoding the first point in space.
    p2: np.ndarray
        The numpy array of shape (3,) encoding the second point in space (vertex).
    p3: np.ndarray
        The numpy array of shape (3,) encoding the third point in space.
    
    Returns
    -------
    float
        The angle (in radiants).
    """
    v1, v2 = p1 - p2, p3 - p2
    u1, u2 = unit_vector(v1), unit_vector(v2)
    
    cos_theta = np.dot(u1, u2)

    return np.arccos(cos_theta)


def dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Computes the dihedral angle (in radiants) between the plane defined by the points `p1`-`p2`-`p3`
    and the one defined by the points `p2`-`p3`-`p4`.

    Arguments
    ---------
    p1: np.ndarray
        The numpy array of shape (3,) encoding the first point in space.
    p2: np.ndarray
        The numpy array of shape (3,) encoding the second point in space.
    p3: np.ndarray
        The numpy array of shape (3,) encoding the third point in space.
    p4: np.ndarray
        The numpy array of shape (3,) encoding the fourth point in space.
    
    Returns
    -------
    float
        The dihedral angle (in radiants).
    """
    # Compute the subsequent vectors connecting the four points
    v1, v2, v3 = p2-p1, p3-p2, p4-p3

    # Check collinearity using cross product
    if np.linalg.norm(np.cross(v1, v2)) < 1e-12:
        raise RuntimeError("Cannot compute dihedral: first three points are collinear.")
    
    if np.linalg.norm(np.cross(v2, v3)) < 1e-12:
        raise RuntimeError("Cannot compute dihedral: last three points are collinear.")
    
    # Compute and normalize the vectors normal to the planes p1-p2-p3 and p2-p3-p4
    n1 = unit_vector(np.cross(v1, v2))
    n2 = unit_vector(np.cross(v2, v3))

    # Normalize the vector along the central bond
    u = unit_vector(v2)

    # Define a vector normal to the plane formed by n1 and u
    m = np.cross(n1, u)

    # Compute the cosine part (cosine of the angle between the normal to the two planes)
    x = np.dot(n1, n2)

    # Compute the sine part (cosine of the angle between the normal to the second plane and
    # the vector on the second plane perpendicular to n1)
    y = np.dot(m, n2)

    return np.arctan2(y, x)
