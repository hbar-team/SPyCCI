import shutil
import subprocess
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from packaging.version import Version, InvalidVersion
from packaging.specifiers import Specifier, SpecifierSet
from os import environ

logger = logging.getLogger(__name__)

###############
# Data models #
###############


@dataclass
class ProgramSpec:
    name: str
    version_marker: Optional[str] = None  # substring to identify the line containing the version
    token_index: Optional[int] = None  # which whitespace token contains the version


@dataclass
class DependencySpec(ProgramSpec):
    versions_map: Optional[Dict[str, List[str]]] = None
    # versions_map: parent_version_prefix -> list of exact allowed dependency versions
    critical: bool = True  # if True, failure to meet this dependency raises an error


@dataclass
class EngineSpec(ProgramSpec):
    dependencies: List[DependencySpec] = field(default_factory=list)


###########################################
# Engine specifications                   #
# Add new engines by extending this list. #
###########################################

ENGINE_SPECS: List[EngineSpec] = [
    EngineSpec(
        name="orca",
        version_marker="Program Version",
        token_index=2,
        dependencies=[
            DependencySpec(
                name="mpirun",
                version_marker="(Open MPI)",
                token_index=3,
                versions_map={
                    "==6.1.*": ["==4.1.6"],
                    "==6.0.*": ["==4.1.6"],
                    "==5.0.*": ["==4.1.1"],
                    "==4.2.*": ["==3.1.4"],
                    "==4.1.*": ["==3.1.3", "==2.1.5"],
                },
                critical=True,
            ),
            DependencySpec(
                name="otool_xtb",
                critical=False,
            ),
        ],
    ),
    EngineSpec(name="xtb", version_marker="xtb version", token_index=3),
    EngineSpec(
        name="crest",
        version_marker="Version",
        token_index=1,
        dependencies=[
            DependencySpec(
                name="xtbiff",
                version_marker="Version",
                token_index=3,
                versions_map={
                    "<3.0": ["==1.1"],   # Only required for crest < 3.0
                },
                critical=True,
            )
        ],
    ),
    EngineSpec(name="dftb+", version_marker="DFTB+ release", token_index=3),
    EngineSpec(name="vmd"),
]

################
# EngineFinder #
################


# Regex to extract the leading numeric version (X, X.Y or X.Y.Z) from a raw token
_VERSION_CORE_RE = re.compile(r"^(\d+(?:\.\d+){0,2})")


def _extract_core_version(token: str) -> str:
    """
    Return the leading numeric version (X, X.Y or X.Y.Z) from a raw token.
    Examples:
      '6.1.0-f.0' -> '6.1.0'
      '3.0.2,'    -> '3.0.2'
      '5.4'       -> '5.4'
      '7'         -> '7'
    """
    m = _VERSION_CORE_RE.match(token)
    if not m:
        raise RuntimeError(f"cannot parse version core from '{token}'")
    return m.group(1)


class EngineFinder:
    def __init__(self, specs: List[EngineSpec]):
        self._specs: Dict[str, EngineSpec] = {s.name: s for s in specs}

    # Public entry point
    def locate(self, name: str, version: Optional[str] = None) -> str:
        if name not in self._specs:
            raise RuntimeError(f"unknown program '{name}'")

        exe = shutil.which(name)
        if not exe:
            raise RuntimeError(f"cannot find '{name}' in the system path")

        spec = self._specs[name]

        if spec.version_marker:
            found_version = self._get_version(spec, exe)
            req_spec = SpecifierSet(version) if version else None
            self._enforce_version(name, found_version, req_spec)

            if spec.dependencies:
                self._check_dependencies(spec, found_version)
        elif version:
            raise RuntimeError(f"version checking not supported for '{name}'")

        return exe

    ####################
    # Internal helpers #
    ####################

    def _get_version(self, spec: ProgramSpec, exe: str) -> Version:
        if not spec.version_marker or spec.token_index is None:
            raise ValueError("spec must define version_marker and token_index to extract version")

        proc = subprocess.run([exe, "--version"], capture_output=True, text=True)
        raw = (proc.stdout or "") + (proc.stderr or "")

        for line in raw.splitlines():
            if spec.version_marker in line:
                raw_token = line.split()[spec.token_index]
                core = _extract_core_version(raw_token)
                break
        else:
            raise RuntimeError(
                f"failed to extract version (version_marker='{spec.version_marker}', index={spec.token_index})"
            )

        try:
            return Version(core)
        except InvalidVersion as e:
            raise RuntimeError(f"Invalid normalized version '{core}' from token '{raw_token}': {e}") from e

    def _enforce_version(self, name: str, found: Version, required: Optional[SpecifierSet]):
        if not required:
            return
        if found not in required:
            raise RuntimeError(f"requested {name} version '{required}' not satisfied by installed '{found}'")

    def _check_dependencies(self, parent_spec: EngineSpec, parent_version: Version):
        for dep in parent_spec.dependencies:
            dep_exe = shutil.which(dep.name)
            if not dep_exe:
                if dep.versions_map:
                    # If dependency is conditional and parent version does NOT match any key, skip silently
                    if not any(parent_version in SpecifierSet(k) for k in dep.versions_map.keys()):
                        continue
                # Unconditional dependency
                if dep.critical:
                    raise RuntimeError(f"{parent_spec.name} requires '{dep.name}', but it was not found in PATH")
                else:
                    logger.warning(f"{parent_spec.name} recommends '{dep.name}', but it was not found in PATH")
                    continue

            dep_version: Optional[Version] = None
            if dep.version_marker and dep.token_index is not None:
                dep_version = self._get_version(dep, dep_exe)

            if dep.versions_map:
                # Only enforce if parent matches one of the specifiers
                matched_parent = False
                for parent_specifier, allowed_spec_list in dep.versions_map.items():
                    if parent_version in SpecifierSet(parent_specifier):
                        matched_parent = True
                        if dep_version is None:
                            raise RuntimeError(
                                f"missing version info for dependency '{dep.name}' required by {parent_spec.name}"
                            )
                        if not any(dep_version in SpecifierSet(spec) for spec in allowed_spec_list):
                            raise RuntimeError(
                                f"{parent_spec.name} {parent_version} requires {dep.name} in "
                                f"[{', '.join(allowed_spec_list)}], found {dep_version}"
                            )
                        break
                # If not matched_parent: dependency not required for this parent version -> skip silently
                continue


##################################################
# Global finder instance + convenience functions #
##################################################

finder = EngineFinder(ENGINE_SPECS)


def locate_orca(path: str = "orca", version: str = "") -> str:
    return finder.locate(path, version=version or None)


def locate_xtb(path: str = "xtb", version: str = "") -> str:
    return finder.locate(path, version=version or None)


def locate_crest(path: str = "crest", version: str = "") -> str:
    return finder.locate(path, version=version or None)


def locate_dftbplus(path: str = "dftb+", version: str = "") -> str:
    return finder.locate(path, version=version or None)


def locate_vmd(path: str = "vmd") -> str:
    return finder.locate(path)


def locate_dftbparamdir() -> str:
    try:
        return environ["DFTBPLUS_PARAM_DIR"]
    except KeyError:
        raise RuntimeError("Failed to locate DFTBPLUS_PARAM_DIR environment variable.")
