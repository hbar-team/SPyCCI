import pytest

import shutil
import subprocess

from spycci.engines.orca import OrcaInput


class FakeProc:
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


def test_OrcaInput___init__(monkeypatch):
    # Fake environment
    outputs = {
        "orca": "Program Version 6.0.1\n",
        "mpirun": "mpirun (Open MPI) 4.1.6\n",
        "otool_xtb": "",
    }

    def fake_which(name):
        base = name.split("/")[-1]
        return f"/fake/bin/{base}"

    def fake_run(cmd, capture_output=True, text=True):
        base = cmd[0].split("/")[-1]
        return FakeProc(stdout=outputs.get(base, ""))

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    # Construct (do not pass ORCAPATH; use default so locate_orca path logic is exercised)
    engine = OrcaInput(
        method="HF",
        basis_set="def2-SVP",
        aux_basis="def2/J",
        solvent="water",
        ORCAPATH="/fake/bin/orca",
    )

    assert engine.method == "HF"
    assert engine.level_of_theory == "OrcaInput || method: HF | basis: def2-SVP | solvent: water"
