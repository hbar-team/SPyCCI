import pytest

from spycci.engines.orca import OrcaInput

ORCADIR = "orcadir"

# Test the OrcaInput class constructor
def test_OrcaInput___init__():

    try:
        engine = OrcaInput(
            method="HF",
            basis_set="def2-SVP",
            aux_basis="def2/J",
            solvent="water",
            ORCADIR=ORCADIR,
        )

    except:
        assert False, "Unenxpected exception raised during OrcaInput class construction"

    else:
        assert engine.method == "HF"
        assert (
            engine.level_of_theory
            == "OrcaInput || method: HF | basis: def2-SVP | solvent: water"
        )
