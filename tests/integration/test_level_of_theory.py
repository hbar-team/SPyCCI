import pytest

from spycci.engines.orca import OrcaInput
from spycci.engines.xtb import XtbInput
from spycci.engines.dftbplus import DFTBInput
from spycci.core.properties import (
    is_orca_level_of_theory,
    is_xtb_level_of_theory,
    is_dftb_level_of_theory,
)


# Test the functions to check the level of theory of the engines
# ------------------------------------------------------------------------------------------

# Test the is_orca_level_of_theory function
def test_is_orca_level_of_theory():

    engine = OrcaInput(
        method="HF",
        basis_set="def2-SVP",
        aux_basis="def2/J",
        solvent="water",
    )

    assert is_orca_level_of_theory(engine.level_of_theory) == True
    assert is_orca_level_of_theory("dummy") == False


# Test the is_xtb_level_of_theory function
def test_XtbInput_is_xtb_level_of_theory():

    engine = XtbInput(
        method="gfn2",
        solvent=None,
    )

    assert is_xtb_level_of_theory(engine.level_of_theory) == True
    assert is_xtb_level_of_theory("dummy") == False


# Test the is_dftb_level_of_theory function
def test_is_dftb_level_of_theory():

    engine = DFTBInput(
        method="DFTB",
        parameters="3ob/3ob-3-1",
        solver=None,
        thirdorder=True,
        dispersion=False,
        fermi=False,
        fermi_temp=300.0,
        parallel="mpi",
        verbose=True,
        DFTBPATH="dummy",
        DFTBPARAMDIR="dummy",
    )

    assert is_dftb_level_of_theory(engine.level_of_theory) == True
    assert is_dftb_level_of_theory("dummy") == False
