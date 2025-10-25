import inspect
import numpy as np

from copy import deepcopy
from typing import Any, Union

def _find_depth(variable: Any) -> int:
    """
    Given an input variable the function computes the number of levels of 
    iterable data. (e.g. a list will have depth 1 while a list of lists a
    depth of 2.)

    Argument
    --------
    variable: Any
        The variable to be evaluated
    
    Returns
    -------
    int
        The depth of the provided object
    """
    depth = 0

    current = deepcopy(variable)
    while True:

        if type(current) == str:
            break

        try:
            buffer = current[0]

        except:
            break

        else:
            depth += 1
            current = buffer

    return depth


def _apply_protection(variable: Any) -> Union[tuple, np.ndarray]:
    """
    Given a generic iterable variable of generic depth, the functions
    returns its read-only version. If a numpy array is encountered,
    its `flags.witeable` attribute is set to `False` if a list or other
    iterable objects are encontered, the content is converted in a tuple.

    Arguments
    ---------
    variable: Any
        The input object to be protected.

    Returns
    -------
    Union[tuple, np.ndarray]
        The read-only version of the protected object.
    """
    if _find_depth(variable) == 0:
        return deepcopy(variable)

    elif isinstance(variable, np.ndarray):
        array = deepcopy(variable)
        array.flags.writeable = False
        return array

    else:
        output = []
        for val in variable:
            protected_val = _apply_protection(val)
            output.append(protected_val)

        return tuple(output)


def protect(function):
    """
    Decorator used to protect the output of a given function from writing
    operations. The iterable output of a function, normally passed by reference,
    is converted into immutable (tuple) or read-only objects (read-only numpy arrays).
    The decorator covers both the case of regolar and generator functions.
    """
    
    # Check if the function will return a generator
    if inspect.isgeneratorfunction(function):
        
        # If yes: yield individually generated protected data 
        def generator_wrapper(*args, **kwargs):
            for item in function(*args, **kwargs):
                yield _apply_protection(item)
        return generator_wrapper
    
    else:

        # If not: define a standalone wrapper to avoid triggering
        # generator mechanics
        def wrapper(*args, **kwargs):
            output = function(*args, **kwargs)
            return _apply_protection(output)
        return wrapper

