import os, shutil, sh
from tempfile import mkdtemp
from spycci.config import get_ncores
from spycci.systems import System, Ensemble
from spycci.tools import split_multixyz
from spycci.tools import cyclization_check
from spycci.tools import process_output
from spycci.core.dependency_finder import locate_crest, locate_executable
import logging

logger = logging.getLogger(__name__)


def tautomer_search(
    mol: System,
    ncores: int = None,
    maxcore=None,
    solvent: str = None,
    remove_tdir: bool = True,
    optionals: str = "",
    CRESTPATH: str = None,
):
    """Tautomer search using CREST.

    Parameters
    ----------
    mol : System object
        input molecule to use in the calculation
    ncores : int, optional
        number of cores, by default all available cores
    maxcore : dummy variable
        dummy variable used for compatibility with Orca calculations
    solvent : str, optional
        ALPB solvent, by default no solvent (vacuum)
    remove_tdir : bool, optional
        temporary work directory will be removed, by default True
    optionals : str, optional
        optional flags for calculation
    CRESTPATH: str, optional
            the path to the crest executable. If set to None (default) the crest executable
            will be loaded automatically.

    Returns
    -------
    tautomers : Ensemble
        Ensemble containing the found tautomers, in order of ascending energy
    """

    if ncores is None:
        ncores = get_ncores()

    if CRESTPATH is None:
        CRESTPATH = locate_crest()

    logger.info(f"{mol.name}, charge {mol.charge} spin {mol.spin} - CREST tautomer search")
    logger.debug(f"Running CREST calculation on {ncores} cores")

    tdir = mkdtemp(prefix=mol.name + "_", suffix="_TAUT", dir=os.getcwd())

    with sh.pushd(tdir):
        mol.geometry.write_xyz("geom.xyz")

        if solvent:
            cmd = f"{CRESTPATH} geom.xyz --alpb {solvent} --chrg {mol.charge} --uhf {mol.spin-1} --mquick --fstrict --tautomerize {optionals} -T {ncores} > output.out 2>> output.err"
            logger.debug(f"Running CREST with command: {cmd}")
            os.system(cmd)

        else:
            cmd = f"{CRESTPATH} geom.xyz --chrg {mol.charge} --uhf {mol.spin-1} --mquick --fstrict --tautomerize {optionals} -T {ncores} > output.out 2>> output.err"
            logger.debug(f"Running CREST with command: {cmd}")
            os.system(cmd)

        if os.path.exists("tautomers.xyz"):
            tautomers_to_check = split_multixyz(mol, file="tautomers.xyz", suffix="t")

            tautomers = []

            while tautomers_to_check:
                tautomer: System = tautomers_to_check.pop(0)
                tautomer.geometry.write_xyz(f"{tautomer.name}.xyz")
                if cyclization_check("geom.xyz", f"{tautomer.name}.xyz") is True:
                    logger.warning(
                        f"Cyclization change spotted for {tautomer.name}, charge {mol.charge} spin {mol.spin}. Removing from list."
                    )
                    mol.flags.append(
                        f"Cyclization change occurred for {tautomer.name} during conformer search. Conformer was removed."
                    )
                else:
                    tautomers.append(tautomer)

            process_output(mol, "CREST", "tautomers", mol.charge, mol.spin)
            if remove_tdir:
                shutil.rmtree(tdir)
            return Ensemble(tautomers)

        else:
            logger.warning(
                f"No tautomers possible for {mol.name}, charge {mol.charge} spin {mol.spin}. Ignoring tautomer search."
            )
            mol.flags.append("No possible tautomers. Tautomer search was ignored.")
            process_output(mol, "CREST", "tautomers", mol.charge, mol.spin)
            if remove_tdir:
                shutil.rmtree(tdir)
            return Ensemble([mol])


def conformer_search(
    mol: System,
    ncores: int = None,
    maxcore=None,
    solvent: str = None,
    remove_tdir: bool = True,
    optionals: str = "",
    CRESTPATH: str = None,
):
    """Conformer search using CREST.

    Parameters
    ----------
    mol : System object
        input molecule to use in the calculation
    ncores : int, optional
        number of cores, by default all available cores
    maxcore : dummy
        dummy variable used for compatibility with Orca calculations
    solvent : str, optional
        ALPB solvent, by default no solvent (vacuum)
    remove_tdir : bool, optional
        temporary work directory will be removed, by default True
    optionals : str, optional
        optional flags for calculation
    CRESTPATH: str, optional
            the path to the crest executable. If set to None (default) the crest executable
            will be loaded automatically.

    Returns
    -------
    conformers : Ensemble
        Ensemble containing the found conformers, in order of ascending energy
    """

    if ncores is None:
        ncores = get_ncores()

    if CRESTPATH is None:
        CRESTPATH = locate_crest()

    logger.info(f"{mol.name}, charge {mol.charge} spin {mol.spin} - CREST conformer search")
    logger.debug(f"Running CREST calculation on {ncores} cores")

    tdir = mkdtemp(prefix=mol.name + "_", suffix="_CONF", dir=os.getcwd())

    with sh.pushd(tdir):
        mol.geometry.write_xyz("geom.xyz")

        if solvent:
            cmd = f"{CRESTPATH} geom.xyz --alpb {solvent} --chrg {mol.charge} --uhf {mol.spin-1} --mquick {optionals} -T {ncores} > output.out 2>> output.err"
            logger.debug(f"Running CREST with command: {cmd}")
            os.system(cmd)

        else:
            cmd = f"{CRESTPATH} geom.xyz --chrg {mol.charge} --uhf {mol.spin-1} --mquick {optionals} -T {ncores} > output.out 2>> output.err"
            logger.debug(f"Running CREST with command: {cmd}")
            os.system(cmd)

        if os.path.exists("crest_conformers.xyz"):
            conformers_to_check = split_multixyz(mol, file="crest_conformers.xyz", suffix="c")

            conformers = []

            while conformers_to_check:
                conformer: System = conformers_to_check.pop(0)
                conformer.geometry.write_xyz(f"{conformer.name}.xyz")
                if cyclization_check("geom.xyz", f"{conformer.name}.xyz") is True:
                    logger.warning(
                        f"Cyclization change spotted for {conformer.name}, charge {mol.charge} spin {mol.spin}. Removing from list."
                    )
                    mol.flags.append(
                        f"Cyclization change occurred for {conformer.name} during conformer search. Conformer was removed."
                    )
                else:
                    conformers.append(conformer)

            process_output(mol, "CREST", "conformers", mol.charge, mol.spin)
            if remove_tdir:
                shutil.rmtree(tdir)
            return Ensemble(conformers)

        else:
            logger.error(
                f"{mol.name}, charge {mol.charge} spin {mol.spin}, conformer search failed. Reverting to original molecule."
            )
            mol.flags.append("Conformer search failed.")
            return Ensemble([mol])


def deprotonate(
    mol: System,
    ncores: int = None,
    maxcore=None,
    solvent: str = None,
    remove_tdir: bool = True,
    optionals: str = "",
    CRESTPATH: str = None,
):
    """Deprotomer search using CREST.

    Parameters
    ----------
    mol : System object
        input molecule to use in the calculation
    ncores : int, optional
        number of cores, by default all available cores
    maxcore : dummy
        dummy variable used for compatibility with Orca calculations
    solvent : str, optional
        ALPB solvent, by default no solvent (vacuum)
    remove_tdir : bool, optional
        temporary work directory will be removed, by default True
    optionals : str, optional
        optional flags for calculation
    CRESTPATH: str, optional
            the path to the crest executable. If set to None (default) the crest executable
            will be loaded automatically.

    Returns
    -------
    deprotomers : Ensemble
        Ensemble containing the found deprotomers, in order of ascending energy
    """

    if ncores is None:
        ncores = get_ncores()

    if CRESTPATH is None:
        CRESTPATH = locate_crest()

    logger.info(f"{mol.name}, charge {mol.charge} spin {mol.spin} - CREST deprotonation")
    logger.debug(f"Running CREST calculation on {ncores} cores")

    tdir = mkdtemp(prefix=mol.name + "_", suffix="_DEPROT", dir=os.getcwd())

    with sh.pushd(tdir):
        mol.geometry.write_xyz("geom.xyz")

        if solvent:
            cmd = f"crest geom.xyz --alpb {solvent} --chrg {mol.charge} --uhf {mol.spin-1} --deprotonate --fstrict {optionals} -T {ncores} > output.out 2>> output.err"
            logger.debug(f"Running CREST with command: {cmd}")
            os.system(cmd)

        else:
            cmd = f"crest geom.xyz --chrg {mol.charge} --uhf {mol.spin-1} --deprotonate --fstrict {optionals} -T {ncores} > output.out 2>> output.err"
            logger.debug(f"Running CREST with command: {cmd}")
            os.system(cmd)

        if os.path.exists("deprotonated.xyz"):
            deprotomers_to_check = split_multixyz(mol, file="deprotonated.xyz", suffix="d", charge=mol.charge - 1)

            deprotomers = []

            while deprotomers_to_check:
                deprotomer: System = deprotomers_to_check.pop(0)
                deprotomer.geometry.write_xyz(f"{deprotomer.name}.xyz")
                if cyclization_check("geom.xyz", f"{deprotomer.name}.xyz") is True:
                    logger.warning(
                        f"Cyclization change spotted for {deprotomer.name}, charge {mol.charge} spin {mol.spin}. Removing from list."
                    )
                    mol.flags.append(
                        f"Cyclization change occurred for {deprotomer.name} during deprotomer search. Deprotomer was removed."
                    )
                else:
                    deprotomers.append(deprotomer)

            process_output(mol, "CREST", "deprotomers", mol.charge, mol.spin)
            if remove_tdir:
                shutil.rmtree(tdir)

            if deprotomers:
                return Ensemble(deprotomers)
            else:
                logger.error(f"{mol.name}, charge {mol.charge} spin {mol.spin}, no suitable deprotomers found.")
                mol.flags.append("No suitable deprotomers.")
                return None

        else:
            logger.error(f"{mol.name}, charge {mol.charge} spin {mol.spin}, deprotomer search failed.")
            mol.flags.append("Deprotomer search failed.")
            return None


def protonate(
    mol: System,
    ncores: int = None,
    maxcore=None,
    solvent: str = None,
    remove_tdir: bool = True,
    optionals: str = "",
    CRESTPATH: str = None,
):
    """Protomer search using CREST.

    Parameters
    ----------
    mol : System object
        input molecule to use in the calculation
    ncores : int, optional
        number of cores, by default all available cores
    maxcore : dummy
        dummy variable used for compatibility with Orca calculations
    solvent : str, optional
        ALPB solvent, by default no solvent (vacuum)
    remove_tdir : bool, optional
        temporary work directory will be removed, by default True
    optionals : str, optional
        optional flags for calculation
    CRESTPATH: str, optional
            the path to the crest executable. If set to None (default) the crest executable
            will be loaded automatically.

    Returns
    -------
    protomers : Ensemble
        Ensemble containing the found protomers, in order of ascending energy
    """

    if ncores is None:
        ncores = get_ncores()

    if CRESTPATH is None:
        CRESTPATH = locate_crest()

    logger.info(f"{mol.name}, charge {mol.charge} spin {mol.spin} - CREST protonation")
    logger.debug(f"Running CREST calculation on {ncores} cores")

    tdir = mkdtemp(prefix=mol.name + "_", suffix="_PROT", dir=os.getcwd())

    with sh.pushd(tdir):
        mol.geometry.write_xyz("geom.xyz")

        if solvent:
            cmd = f"crest geom.xyz --alpb {solvent} --chrg {mol.charge} --uhf {mol.spin-1} --protonate --fstrict {optionals} -T {ncores} > output.out 2>> output.err"
            logger.debug(f"Running CREST with command: {cmd}")
            os.system(cmd)

        else:
            cmd = f"crest geom.xyz --chrg {mol.charge} --uhf {mol.spin-1} --protonate --fstrict {optionals} -T {ncores} > output.out 2>> output.err"
            logger.debug(f"Running CREST with command: {cmd}")
            os.system(cmd)

        if os.path.exists("protonated.xyz"):
            protomers_to_check = split_multixyz(mol, file="protonated.xyz", suffix="p", charge=mol.charge + 1)

            protomers = []

            while protomers_to_check:
                protomer: System = protomers_to_check.pop(0)
                protomer.geometry.write_xyz(f"{protomer.name}.xyz")
                if cyclization_check("geom.xyz", f"{protomer.name}.xyz") is True:
                    logger.warning(
                        f"Cyclization change spotted for {protomer.name}, charge {mol.charge} spin {mol.spin}. Removing from list."
                    )
                    mol.flags.append(
                        f"Cyclization change occurred for {protomer.name} during deprotomer search. Protomer was removed."
                    )
                else:
                    protomers.append(protomer)

            process_output(mol, "CREST", "protomers", mol.charge, mol.spin)
            if remove_tdir:
                shutil.rmtree(tdir)

            if protomers:
                return Ensemble(protomers)
            else:
                logger.error(f"{mol.name}, charge {mol.charge} spin {mol.spin}, no suitable protomers found.")
                mol.flags.append("No suitable protomers.")
                return None
        else:
            logger.error(f"{mol.name}, charge {mol.charge} spin {mol.spin}, protomer search failed.")
            mol.flags.append("Protomer search failed.")
            return None


def qcg_grow(
    solute: System,
    solvent: System,
    charge: int = None,
    spin: int = None,
    method: str = "gfn2",
    nsolv: int = 0,
    ncores: int = None,
    maxcore=None,
    alpb_solvent: str = None,
    optionals: str = "",
    remove_tdir: bool = True,
    CRESTPATH: str = None,
):
    """Quantum Cluster Growth using CREST.

    Parameters
    ----------
    solute : System object
        solute molecule to use in the calculation
    solvent : System object
        solvent molecule to use in the calculation
    charge : int, optional
        total charge of the system. Default is taken from the solute molecule.
    spin : int, optional
        total spin of the system. Default is taken from the solute molecule.
    method : str
        method for the geometry optimizations, by default gfn2
        Alternative options: gfn1, gfnff
    nsolv : int
        number of solvent molecules to add to the cluster, by default 0 (unconstrained).
        If a number is not specified, the program will keep adding solvent
        molecules until convergence is reached, or 150 molecules are added.
    ncores : int, optional
        number of cores, by default all available cores
    maxcore : dummy
        dummy variable used for compatibility with Orca calculations
    alpb_solvent : str, optional
        ALPB solvent, by default no solvent (vacuum)
    optionals : str, optional
        optional flags for calculation
    remove_tdir : bool, optional
        temporary work directory will be removed, by default True
    CRESTPATH: str, optional
            the path to the crest executable. If set to None (default) the crest executable
            will be loaded automatically.

    Returns
    -------
    cluster : System object
        System object containing the explicitly solvated input molecule
    """

    try:
        locate_executable("xtbiff")
    except:
        logger.warning("xtbiff executable not found. Cannot continue with QCG run.")
        return None

    if ncores is None:
        ncores = get_ncores()

    if CRESTPATH is None:
        CRESTPATH = locate_crest()

    if charge is None:
        charge = solute.charge
    if spin is None:
        spin = solute.spin

    logger.info(f"{solute.name}, charge {charge} spin {spin} - CREST QCG GROW - {nsolv} solvent molecules")
    logger.debug(f"Running CREST calculation on {ncores} cores")

    tdir = mkdtemp(prefix=solute.name + "_", suffix="_QCG_G", dir=os.getcwd())

    with sh.pushd(tdir):
        solute.geometry.write_xyz("solute.xyz")
        solvent.geometry.write_xyz("solvent.xyz")

        if alpb_solvent:
            cmd = f"crest solute.xyz --qcg solvent.xyz --nsolv {nsolv} --{method} --alpb {alpb_solvent} --chrg {charge} --uhf {spin-1} {optionals} --T {ncores} > output.out 2>> output.err"
            logger.debug(f"Running CREST with command: {cmd}")
            os.system(cmd)

        else:
            cmd = f"crest solute.xyz --qcg solvent.xyz --nsolv {nsolv} --{method} --chrg {charge} --uhf {spin-1} {optionals} --T {ncores} > output.out 2>> output.err"
            logger.debug(f"Running CREST with command: {cmd}")
            os.system(cmd)

        solute.geometry.write_xyz(f"{solute.name}.xyz")
        cluster = System.from_xyz(f"{solute.name}.xyz", charge=charge, spin=spin)

        try:
            cluster.geometry.load_xyz("grow/cluster.xyz")
        except:
            logger.error(f"{solute.name}, charge {solute.charge} spin {solute.spin}, cluster growth failed.")
            solute.flags.append("Cluster growth failed.")
            return None

        process_output(solute, "QCG", "grow", charge=charge, spin=spin)
        if remove_tdir:
            shutil.rmtree(tdir)

        return cluster


def qcg_ensemble(
    solute: System,
    solvent: System,
    charge: int = None,
    spin: int = None,
    method: str = "gfn2",
    enslvl: str = "gfn2",
    ensemble_choice: str = "full_ensemble",
    nsolv: int = 0,
    ncores: int = None,
    maxcore=None,
    alpb_solvent: str = None,
    optionals: str = "",
    remove_tdir: bool = True,
    CRESTPATH: str = None,
):
    """Quantum Cluster Growth + ensemble generation using CREST.

    Parameters
    ----------
    solute : System object
        solute molecule to use in the calculation
    solvent : System object
        solvent molecule to use in the calculation
    charge : int, optional
        total charge of the molecule. Default is taken from the solute molecule.
    spin : int, optional
        total spin of the molecule. Default is taken from the solute molecule.
    method : str
        method for the geometry optimizations, by default gfn2
        Alternative options: gfn1, gfnff
    enslvl : str
        method for the ensemble optimization, by default gfn2
        Alternative options: gfn1, gfnff
    ensemble_choice : str
        file containing the chosen ensemble after generation, by default "full_ensemble". Available
        options are:
        - "full_ensemble"
        - "final_ensemble"
        - "crest_best"
    nsolv : int
        number of solvent molecules to add to the cluster, by default 0 (unconstrained).
        If a number is not specified, the program will keep adding solvent
        molecules until convergence is reached, or 150 molecules are added.
    ncores : int, optional
        number of cores, by default all available cores
    maxcore : dummy
        dummy variable used for compatibility with Orca calculations
    alpb_solvent : str, optional
        ALPB solvent, by default no solvent (vacuum)
    optionals : str, optional
        optional flags for calculation
    remove_tdir : bool, optional
        temporary work directory will be removed, by default True
    CRESTPATH: str, optional
            the path to the crest executable. If set to None (default) the crest executable
            will be loaded automatically.

    Returns
    -------
    ensemble : Ensemble
        Ensemble object containing the explicitly solvated input molecule, with updated energy
        coming from enseble generation (electronic contribution only). The vibrational contribution
        is taken from the input solute molecule (if present), while the electronic contribution
        is taken as the weighted average of all generated ensembles.
    """

    try:
        locate_executable("xtbiff")
    except:
        logger.warning("xtbiff executable not found. Cannot continue with QCG run.")
        return None

    if ncores is None:
        ncores = get_ncores()

    if CRESTPATH is None:
        CRESTPATH = locate_crest()

    if charge is None:
        charge = solute.charge
    if spin is None:
        spin = solute.spin

    logger.info(f"{solute.name}, charge {charge} spin {spin} - CREST QCG ENSEMBLE - {nsolv} solvent molecules")
    logger.debug(f"Running CREST calculation on {ncores} cores")

    tdir = mkdtemp(prefix=solute.name + "_", suffix="_QCG_E", dir=os.getcwd())

    with sh.pushd(tdir):
        solute.geometry.write_xyz("solute.xyz")
        solvent.geometry.write_xyz("solvent.xyz")

        if alpb_solvent:
            cmd = f"crest solute.xyz --qcg solvent.xyz --nsolv {nsolv} --{method} --ensemble --enslvl {enslvl} --alpb {alpb_solvent} --chrg {charge} --uhf {spin-1} {optionals} --T {ncores} > output.out 2>> output.err"
            logger.debug(f"Running CREST with command: {cmd}")
            os.system(cmd)

        else:
            cmd = f"crest solute.xyz --qcg solvent.xyz --nsolv {nsolv} --{method} --ensemble --enslvl {enslvl} --chrg {charge} --uhf {spin-1} {optionals} --T {ncores} > output.out 2>> output.err"
            logger.debug(f"Running CREST with command: {cmd}")
            os.system(cmd)

        try:
            ensemble = split_multixyz(solute, file=f"ensemble/{ensemble_choice}.xyz", suffix="e")

        except:
            logger.error(f"{solute.name}, charge {solute.charge} spin {solute.spin}, cluster growth failed.")
            solute.flags.append("Cluster growth failed.")
            return None

        process_output(solute, "QCG", "ensemble", charge=charge, spin=spin)
        if remove_tdir:
            shutil.rmtree(tdir)

        return Ensemble(ensemble)
