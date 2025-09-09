import pytest

import shutil
import subprocess

from spycci.engines.xtb import XtbInput


class FakeProc:
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


# Test the XtbInput class constructor
def test_XtbInput___init__(monkeypatch):

    # Fake environment
    outputs = {
        "xtb": "* xtb version 6.6.1\n",
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
        engine = XtbInput(
            method="gfn2",
            solvent="water",
            XTBPATH="/fake/bin/xtb",
        )

    except:
        assert False, "Unenxpected exception raised during XtbInput class construction"

    else:
        assert engine.method == "gfn2"
        assert engine.level_of_theory == "XtbInput || method: gfn2 | solvent: water"
