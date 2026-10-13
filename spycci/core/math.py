import inspect
import numpy as np
from functools import wraps


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


def validate_3d_input(function):
    """
    Convert and validate arguments annotated as np.ndarray.

    The decorator converts the arguments annotated as np.ndarray to the correct
    format using np.asarray() and verifies that each array has shape (3,), 
    corresponding to a three-dimensional vector. Both positional and keyword arguments
    are supported. Parameters without an explicit np.ndarray annotation are left unchanged.

    Raises
    ------
    ValueError
        If an annotated argument does not have shape (3,).
    """
    
    # Obtain target function signature
    signature = inspect.signature(function)

    # Inspect function signature for the names of the parameters annoted as numpy arrays
    array_names = {name for name, param in signature.parameters.items() if param.annotation is np.ndarray}

    @wraps(function)
    def wrapped(*args, **kwargs):

        # Bound the args and kwargs to the function signature to avoid conversion issues
        # when user passes args as kwargs - e.g. func(x, y) called as func(p1=x, p2=y)
        bound = signature.bind(*args, **kwargs)

        for name, value in bound.arguments.items():

            # If the argument is not expected to be an array continue
            if name not in array_names:
                continue

            # Convert lists to numpy arrays and verify the shape to be (3,)
            array = np.asarray(value)

            if array.shape != (3,):
                raise ValueError(f"Argument '{name}' must have shape (3,), got {array.shape}.")

            # Replace the original arguments with numpy arrays
            bound.arguments[name] = array

        return function(*bound.args, **bound.kwargs)

    return wrapped



@validate_3d_input
def distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
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


@validate_3d_input
def angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Computes the angle (in radians) formed by the points (`p1`, `p2` and `p3`) in the 3D
    space where `p2` represents the vertex point.

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
        The angle (in radians).
    """
    v1, v2 = p1 - p2, p3 - p2
    u1, u2 = unit_vector(v1), unit_vector(v2)
    
    cos_theta = np.dot(u1, u2)

    return np.arccos(cos_theta)


@validate_3d_input
def dihedral_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Computes the dihedral angle (in radians) in the 3D space between the plane defined
    by the points `p1`-`p2`-`p3` and the one defined by the points `p2`-`p3`-`p4`.

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
        The dihedral angle (in radians).
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
  
    # Define a vector parallel to u from the cross product of the normals
    m = np.cross(n1, n2)

    # Compute the cosine part (cosine of the angle between the normal to the two planes)
    x = np.dot(n1, n2)

    # Compute the sine part (positive sign if m and u point in the same direction)
    u = unit_vector(v2)
    y = np.dot(m, u)

    return np.arctan2(y, x)
