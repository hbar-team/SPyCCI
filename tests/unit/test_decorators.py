import pytest
import numpy as np

from numpy.testing import assert_array_almost_equal, assert_almost_equal

from typing import Any

from spycci.core.decorators import protect

@protect
def mirror_function(argument: Any) -> Any:
    "A simple mirror function returning the same argument it was provided with"
    return argument


# General test for the @protect decorator

def test_protect_number():
    assert_almost_equal(mirror_function(1.), 1., decimal=6)

def test_protect_string():
    assert mirror_function("test") == "test"

def test_protect_list():
    assert mirror_function(["a", "b", "c"]) == ("a", "b", "c")

def test_protect_list_of_lists():
    assert mirror_function([["a", "b"], ["c", "d"]]) == (("a", "b"), ("c", "d"))

def test_protect_tuple():
    assert mirror_function(("a", "b", "c")) == ("a", "b", "c")

def test_protect_np_array():

    value = mirror_function(np.array([1., 2., 3.]))

    assert type(value) == np.ndarray
    assert value.flags.writeable == False

def test_protect_list_of_np_array():

    value = mirror_function([np.array([1., 2., 3.]), np.array([4., 5., 6.])])

    assert type(value) == tuple
    for v in value:
        assert type(v) == np.ndarray
        assert v.flags.writeable == False
