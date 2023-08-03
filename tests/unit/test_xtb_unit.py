import pytest

from spycci.engines.xtb import XtbInput

XTBPATH = "xtbpath"

# Test the XtbInput class constructor
def test_XtbInput___init__():

    try:
        engine = XtbInput(
            method="gfn2",
            solvent="water",
            XTBPATH=XTBPATH,
        )

    except:
        assert False, "Unenxpected exception raised during XtbInput class construction"

    else:
        assert engine.method == "gfn2"
        assert engine.level_of_theory == "XtbInput || method: gfn2 | solvent: water"
