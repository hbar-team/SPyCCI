import logging
import pytest
from spycci.core import dependency_finder as df


class FakeProc:
    def __init__(self, stdout="", stderr=""):
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def exec_env(monkeypatch):
    """
    Configure a fake execution environment for dependency_finder.
    Call the returned function with desired parameters before using locate_* APIs.

    Parameters
    ----------
    outputs: dict[str, str]
        dict name -> '--version' combined stdout (line-based)
    missing: set[str]
        executable basenames to hide
    extra_present: set[str]
        additional basenames to report present (i.e., 'otool_xtb' for orca), is overridden by `missing`
    path_map: dict[str, str]
        dict name -> custom absolute path (overrides default /usr/bin/<name>)

    Returns
    -------
    configure: function
        function to configure the fake environment
    """
    state = {
        "outputs": {},
        "missing": set(),
        "extra_present": {"otool_xtb"},  # optional link for orca
        "path_map": {},
    }

    def configure(
        *,  # force keyword args
        outputs=None,
        missing=None,
        extra_present=None,
        path_map=None,
    ):
        if outputs is not None:
            state["outputs"] = outputs
        if missing is not None:
            state["missing"] = set(missing)
        if extra_present is not None:
            state["extra_present"] = set(extra_present)
        if path_map is not None:
            state["path_map"] = path_map

        def fake_which(name):
            base = name.split("/")[-1]
            if base in state["missing"]:
                return None
            if base in state["path_map"]:
                return state["path_map"][base]
            if base in state["outputs"] or base in state["extra_present"]:
                return f"/usr/bin/{base}"
            return None

        def fake_run(cmd, capture_output=True, text=True):
            base = cmd[0].split("/")[-1]
            out = state["outputs"].get(base, "")
            return FakeProc(stdout=out, stderr="")

        monkeypatch.setattr(df.shutil, "which", fake_which)
        monkeypatch.setattr(df.subprocess, "run", fake_run)
        return state

    return configure


########################
# Core version parsing #
########################


def test_extract_core_version_valid():
    assert df._extract_core_version("6.1.0-f.0") == "6.1.0"
    assert df._extract_core_version("3.0.2,") == "3.0.2"
    assert df._extract_core_version("5.4") == "5.4"
    assert df._extract_core_version("7") == "7"


def test_extract_core_version_invalid():
    with pytest.raises(RuntimeError):
        df._extract_core_version("abc")
    with pytest.raises(RuntimeError):
        df._extract_core_version("")


########################
# Unknown program name #
########################


def test_unknown_program():
    with pytest.raises(RuntimeError, match="unknown program"):
        df.finder.locate("nonexistent_program")


################################
# Program without version info #
################################


def test_vmd_no_version(exec_env):
    exec_env(outputs={"vmd": ""})
    path = df.locate_vmd()
    assert path.endswith("/vmd")
    with pytest.raises(RuntimeError, match="version checking not supported"):
        df.finder.locate("vmd", version="1.0")


#############################
# Basic version enforcement #
#############################


def test_xtb_version_exact_success(exec_env):
    exec_env(outputs={"xtb": "* xtb version 6.5.1\n"})
    assert df.locate_xtb(version="6.5.1").endswith("/xtb")


def test_xtb_version_mismatch(exec_env):
    exec_env(outputs={"xtb": "* xtb version 6.5.1\n"})
    with pytest.raises(RuntimeError, match="requested xtb version"):
        df.locate_xtb(version="6.5.2")


def test_xtb_version_specifier(exec_env):
    exec_env(outputs={"xtb": "* xtb version 6.5.1\n"})
    assert df.locate_xtb(version=">=6.0,<7.0").endswith("/xtb")
    with pytest.raises(RuntimeError):
        df.locate_xtb(version=">=7.0")


########################################
# ORCA + mandatory dependency (mpirun) #
########################################


def test_orca_dependency_success(exec_env):
    exec_env(
        outputs={
            "orca": "Program Version 6.1.0-f.0\n",
            "mpirun": "mpirun (Open MPI) 4.1.6\n",
        },
        missing={"otool_xtb"},
    )
    path = df.locate_orca(version="6.1.*")
    assert path.endswith("/orca")


def test_orca_dependency_mismatch(exec_env):
    exec_env(
        outputs={
            "orca": "Program Version 6.0.1\n",
            "mpirun": "mpirun (Open MPI) 4.1.5\n",
        }
    )
    with pytest.raises(RuntimeError, match="requires mpirun"):
        df.locate_orca(version="6.0.*")


def test_orca_optional_dependency_warning(exec_env, caplog):
    exec_env(
        outputs={
            "orca": "Program Version 6.0.1\n",
            "mpirun": "mpirun (Open MPI) 4.1.6\n",
        },
        missing={"otool_xtb"},
    )
    caplog.set_level(logging.WARNING)
    df.locate_orca()
    assert any("otool_xtb" in rec.message for rec in caplog.records)


################################
# CREST conditional dependency #
################################


def test_crest_requires_xtbiff_before_v3(exec_env):
    exec_env(
        outputs={
            "crest": "Version 2.12\n",
            "xtbiff": "| 2016-17, Version 1.1 |\n",
        }
    )
    assert df.locate_crest(version="<3.0").endswith("/crest")


def test_crest_requires_xtbiff_missing(exec_env):
    exec_env(outputs={"crest": "Version 2.12\n"}, missing={"xtbiff"})
    with pytest.raises(RuntimeError, match="requires 'xtbiff'"):
        df.locate_crest(version="<3.0")


def test_crest_requires_xtbiff_version_mismatch(exec_env):
    exec_env(
        outputs={
            "crest": "Version 2.12\n",
            "xtbiff": "| 2016-17, Version 1.0 |\n",
        }
    )
    with pytest.raises(RuntimeError, match="requires xtbiff"):
        df.locate_crest()


def test_crest_after_v3_xtbiff_not_required(exec_env):
    exec_env(outputs={"crest": "Version 3.0.1\n"}, missing={"xtbiff"})
    assert df.locate_crest(version=">=3.0").endswith("/crest")


########################################
# Failure extracting / parsing version #
########################################


def test_missing_version_marker(exec_env):
    exec_env(outputs={"xtb": "some unrelated output\n"})
    with pytest.raises(RuntimeError, match="failed to extract version"):
        df.locate_xtb()


def test_invalid_version_token(exec_env):
    exec_env(
        outputs={
            "orca": "Program Version notaversion\n",
            "mpirun": "mpirun (Open MPI) 4.1.6\n",
        }
    )
    with pytest.raises(RuntimeError, match="cannot parse version core"):
        df.locate_orca()


#################################
# Path with directory component #
#################################


def test_locate_with_path_component(exec_env):
    exec_env(
        outputs={
            "orca": "Program Version 6.0.1\n",
            "mpirun": "mpirun (Open MPI) 4.1.6\n",
        },
        path_map={"orca": "/custom/bin/orca"},
    )
    assert df.finder.locate("/custom/bin/orca").endswith("/orca")


##############################
# Missing executable in PATH #
##############################


def test_missing_executable(exec_env):
    exec_env(outputs={}, missing={"xtb"})
    with pytest.raises(RuntimeError, match="cannot find 'xtb'"):
        df.locate_xtb()


###############################################
# Invalid normalized version after extraction #
###############################################


def test_orca_invalid_normalized_version(exec_env, monkeypatch):
    """
    Force the code path that raises:
      Invalid normalized version '{core}' from token '{raw_token}'
    by monkeypatching _extract_core_version to return an invalid PEP 440 string.
    """
    exec_env(
        outputs={
            "orca": "Program Version 6.1.0-f.0\n",
            "mpirun": "mpirun (Open MPI) 4.1.6\n",
        }
    )

    def bad_core(_token: str) -> str:
        # Invalid PEP 440 version (double dot)
        return "1..2"

    monkeypatch.setattr(df, "_extract_core_version", bad_core)
    with pytest.raises(RuntimeError, match="Invalid normalized version"):
        df.locate_orca()


############################################################
# Dependency required but no version markers (forced case) #
############################################################


def test_dependency_missing_version_info(exec_env):
    exec_env(
        outputs={
            "crest": "Version 2.12\n",
            "xtbiff": "| 2016-17, Version 1.1 |\n",
        }
    )
    crest_spec = df.finder._specs.get("crest")
    xtbiff_dep = None
    for dep in getattr(crest_spec, "dependencies", []):
        if dep.name == "xtbiff":
            xtbiff_dep = dep
            break
    if xtbiff_dep is None:
        pytest.skip("xtbiff dependency not defined; update test if spec changes")
    original_marker = getattr(xtbiff_dep, "version_marker", None)
    original_index = getattr(xtbiff_dep, "token_index", None)
    xtbiff_dep.version_marker = None
    xtbiff_dep.token_index = None
    try:
        with pytest.raises(RuntimeError, match="missing version info for dependency 'xtbiff'"):
            df.locate_crest(version="<3.0")
    finally:
        xtbiff_dep.version_marker = original_marker
        xtbiff_dep.token_index = original_index


###########################
# locate_dftbplus wrapper #
###########################


def test_locate_dftbplus(exec_env):
    exec_env(outputs={"dftb+": "|  DFTB+ release 23.1\n"})
    assert df.locate_dftbplus(version="23.1").endswith("/dftb+")


######################################
# locate_dftbparamdir (env variable) #
######################################


def test_locate_dftbparamdir_success(monkeypatch):
    monkeypatch.setenv("DFTBPLUS_PARAM_DIR", "/fake/params")
    assert df.locate_dftbparamdir() == "/fake/params"


def test_locate_dftbparamdir_failure(monkeypatch):
    monkeypatch.delenv("DFTBPLUS_PARAM_DIR", raising=False)
    with pytest.raises(RuntimeError, match="DFTBPLUS_PARAM_DIR"):
        df.locate_dftbparamdir()
