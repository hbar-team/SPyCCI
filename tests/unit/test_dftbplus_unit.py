import pytest

import shutil
import subprocess

from spycci.engines.dftbplus import DFTBInput


class FakeProc:
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


# Test the DFTBInput class constructor
def test_DFTBInput___init__(monkeypatch):

    # Fake environment
    outputs = {
        "dftb+": "|  DFTB+ release 23.1\n",
    }

    def fake_which(name):
        base = name.split("/")[-1]
        return f"/fake/bin/{base}"

    def fake_run(cmd, capture_output=True, text=True):
        base = cmd[0].split("/")[-1]
        return FakeProc(stdout=outputs.get(base, ""))

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

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
            DFTBPATH="/fake/bin/dftb+",
            DFTBPARAMDIR="/fake/params",
        )

    except:
        assert False, "Unenxpected exception raised during DFTBInput class construction"

    else:
        assert engine.method == "DFTB"
        assert (
            engine.level_of_theory
            == "DFTBInput || method: DFTB | parameters: 3ob/3ob-3-1 | 3rd order: True | dispersion: False"
        )
