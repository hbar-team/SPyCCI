import pytest

from spycci.engines.dftbplus import DFTBInput

DFTBPLUS = "dftbplus"
DFTBPARAMDIR = "dftbparamdir"

# Test the DFTBInput class constructor
def test_DFTBInput___init__():

    try:
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
            DFTBPATH=DFTBPLUS,
            DFTBPARAMDIR=DFTBPARAMDIR,
        )

    except:
        assert False, "Unenxpected exception raised during DFTBInput class construction"

    else:
        assert engine.method == "DFTB"
        assert (
            engine.level_of_theory
            == "DFTBInput || method: DFTB | parameters: 3ob/3ob-3-1 | 3rd order: True | dispersion: False"
        )
