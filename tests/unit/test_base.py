import pytest

from spycci.core.base import Engine

# Test the Engine class constructor
def test_Engine___init__():

    try:
        obj = Engine("DummyMethod")
    
    except:
        assert False, "Unenxpected exception raised during Engine class construction"
    
    else:    
        assert obj.method == "DummyMethod"
        assert obj.level_of_theory == "Engine || method: DummyMethod"


# Test that, when inheriting form Engine, the expected behavior is obtained
def test_Engine_inheritance():

    class DerivedEngine(Engine):

        def __init__(self, method: str) -> None:
            super().__init__(method)
    
    try:
        dobj = DerivedEngine("DerivedMethod")
    
    except:
        assert False, "Unenxpected exception raised during inheritance from Engine"
    
    else:    
        assert dobj.method == "DerivedMethod"
        assert dobj.level_of_theory == "DerivedEngine || method: DerivedMethod"