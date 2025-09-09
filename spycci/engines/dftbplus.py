import os, copy, shutil, sh

import spycci.config as cfg
from spycci.config import get_ncores
from os.path import join
from tempfile import mkdtemp
from spycci.core.base import Engine
from spycci.systems import System, Ensemble
from spycci.tools import (
    process_output,
    save_dftb_trajectory,
    compress_dftb_trajectory,
    split_multixyz,
)
from spycci.tools.internaltools import clean_suffix

from typing import Dict
from spycci.core.dependency_finder import locate_dftbplus, locate_dftbparamdir

import logging

logger = logging.getLogger(__name__)


class DFTBInput(Engine):
    """
    Interface for running DFTB+ calculations

    Parameters
    ----------
    method : str
        level of theory, by default "DFTB". "xTB" also supported.
    parameters : str
        parameters to be used for the DFTB Hamiltonian (by default 3ob)
    solver : str
        LAPACK eigensolver method (check manual for available options)
    thirdorder : bool
        activates the 3rd order terms in the DFTB Hamiltonian
    dispersion : bool
        activates D3 dispersion corrections (off by default)
    fermi: bool
        Fills the single particle levels according to a Fermi distribution (off by
        default).
    fermi_temp: float
        Electronic temperature in Kelvin units. Note, this is ignored for thermostated
        simulations. By default, 300 K.
    parallel : str
        selects either openmpi-parallel version (mpi) or shared memory version (nompi)
    verbose : bool
        if set to True, saves the full DFTB+ output, otherwise, only the smaller files
    DFTBPATH: str
        the path to the dftb+ executable. If set to None (default) the dftb+ executable
        will be loaded automatically.
    DFTBPARAMDIR: str
        the path to the DFTBPLUS_PARAM_DIR environment variable, where the Slater-Koster
        files are located

    Attributes
    ----------
    method : str
        level of theory, by default "DFTB". "xTB" also supported.
    parameters : str
        parameters to be used for the DFTB Hamiltonian (by default 3ob)
    solver : str
        LAPACK eigensolver method (check manual for available options)
    dispersion : bool
        activates D3 dispersion corrections (off by default)
    parallel : str
        selects either openmpi-parallel version (mpi) or shared memory version (nompi)
    verbose : bool
        if set to True, saves the full DFTB+ output, otherwise, only the smaller files
    """

    def __init__(
        self,
        method: str = "DFTB",
        parameters: str = "3ob/3ob-3-1",
        solver: str = None,
        thirdorder: bool = True,
        dispersion: bool = False,
        fermi: bool = False,
        fermi_temp: float = 300.0,
        parallel: str = "mpi",
        verbose: bool = True,
        DFTBPATH: str = None,
        DFTBPARAMDIR: str = None,
    ) -> None:
        super().__init__(method)

        self.parameters = parameters
        self.solver = solver
        self.thirdorder = thirdorder
        self.dispersion = dispersion
        self.fermi = fermi
        self.fermi_temp = fermi_temp
        self.parallel = parallel
        self.verbose = verbose  # add to docs
        if self.verbose:
            self.output_path = "output.out"
        else:
            self.output_path = "/dev/null"

        self.__DFTBPATH = locate_dftbplus(DFTBPATH if DFTBPATH else "dftb+")
        self.__DFTBPARAMDIR = DFTBPARAMDIR if DFTBPARAMDIR else locate_dftbparamdir()

        self.level_of_theory += f" | parameters: {parameters} | 3rd order: {thirdorder} | dispersion: {dispersion}"

        self.__output_suffix = "DFTB"
        self.__output_suffix += "3" if thirdorder else ""
        self.__output_suffix += "-D3" if dispersion else ""
        self.__output_suffix = clean_suffix(self.__output_suffix)

        self.atom_dict = {
            "Br": "d",
            "C": "p",
            "Ca": "p",
            "Cl": "d",
            "F": "p",
            "H": "s",
            "I": "d",
            "K": "p",
            "Mg": "p",
            "N": "p",
            "Na": "p",
            "O": "p",
            "P": "d",
            "S": "d",
            "Zn": "d",
        }

        self.hubbard_derivs = {
            "Br": -0.0573,
            "C": -0.1492,
            "Ca": -0.0340,
            "Cl": -0.0697,
            "F": -0.1623,
            "H": -0.1857,
            "I": -0.0433,
            "K": -0.0339,
            "Mg": -0.02,
            "N": -0.1535,
            "Na": -0.0454,
            "O": -0.1575,
            "P": -0.14,
            "S": -0.11,
            "Zn": -0.03,
        }

        self.spin_constants = {
            "H": [-0.072],
            "C": [-0.031, -0.025, -0.025, -0.023],
            "N": [-0.033, -0.027, -0.027, -0.026],
            "O": [-0.035, -0.030, -0.030, -0.028],
            "S": [-0.021, -0.017, 0.000, -0.017, -0.016, 0.000, 0.000, 0.000, -0.080],
        }

    def write_input(
        self,
        mol: System,
        job_info: Dict,
    ) -> None:
        mol.write_gen(f"{mol.name}.gen")

        with open(f"{mol.name}.gen") as file:
            lines = file.readlines()
            atom_types = lines[1].split()

        input = "Geometry = GenFormat {\n" f'  <<< "{mol.name}.gen"\n' "}\n\n"

        if job_info["type"] == "spe":
            input += "Driver = GeometryOptimization {\n" "  MaxSteps = 0\n" "}\n\n"

        elif job_info["type"] == "opt":
            input += (
                "Driver = GeometryOptimization {\n"
                f"  LatticeOpt = {'Yes' if job_info['latticeopt'] else 'No'}\n"
                "}\n\n"
            )

        elif job_info["type"] == "md_nvt":
            input += (
                "Driver = VelocityVerlet {\n"
                f"  TimeStep [fs] = {job_info['timestep']}\n"
                "  Thermostat = NoseHoover {\n"
                f"    Temperature [K] = {job_info['temperature']}\n"
                "    CouplingStrength [cm^-1] = 3200\n"
                "  }\n"
                f"  Steps = {job_info['steps']}\n"
                "  MovedAtoms = 1:-1\n"
                f"  MDRestartFrequency = {job_info['mdrestartfreq']}\n"
            )
            ### --> VELOCITIES HAVE BEEN REMOVED IN THE LATEST VERSION
            # input += "  Velocities [AA/ps] {\n"
            # for velocity in mol.velocities:
            #     input += f"    {velocity[1:]}"
            # input += "  }\n"
            ### <--
            input += "}\n\n"

        elif job_info["type"] == "simulated_annealing":
            input += (
                "Driver = VelocityVerlet {\n"
                f"  TimeStep [fs] = {job_info['timestep']}\n"
                "  Thermostat = NoseHoover {\n"
                "    Temperature [Kelvin] = TemperatureProfile {\n"
                f"      constant 1 {job_info['start_temp']}\n"
                f"      linear {job_info['ramp_steps']-1} {job_info['target_temp']}\n"
                f"      constant {job_info['hold_steps']} {job_info['target_temp']}\n"
                f"      linear {job_info['ramp_steps']} {job_info['start_temp']}\n"
                "    }\n"
                "    CouplingStrength [cm^-1] = 3200\n"
                "  }\n"
                "  MovedAtoms = 1:-1\n"
                f"  MDRestartFrequency = {job_info['mdrestartfreq']}\n"
            )
            ### --> VELOCITIES HAVE BEEN REMOVED IN THE LATEST VERSION
            # input += "  Velocities [AA/ps] {\n"
            # for velocity in mol.velocities:
            #     input += f"    {velocity[1:]}"
            # input += "  }\n"
            ### <--
            input += "}\n" "\n"

        input += f"Hamiltonian = {self.method} {{\n" "  MaxSCCIterations = 500\n" f"  Charge = {mol.charge}\n"

        if self.fermi:
            input += "  Filling = Fermi {\n" f"    Temperature [K] = {self.fermi_temp}\n" "  }\n"

        if mol.spin != 1:
            input += (
                "  SpinPolarisation = Colinear {\n"
                f"    UnpairedElectrons = {mol.spin-1}\n"
                "  }\n"
                "  SpinConstants = {\n"
            )
            if self.method == "DFTB":
                input += "    ShellResolvedSpin = Yes\n"
            for atom in atom_types:
                input += (
                    f"    {atom} = {{\n"
                    f"      {' '.join(str(spin) for spin in self.spin_constants[atom])}\n"
                    "    }\n"
                )
            input += "  }\n"

        if self.method == "DFTB":
            if self.solver:
                input += f"  Solver = {self.solver} {{}}\n"
            input += (
                "  Scc = Yes\n"
                "  SlaterKosterFiles = Type2FileNames {\n"
                f'    Prefix = "{join(self.__DFTBPARAMDIR, self.parameters)}/"\n'
                '    Separator = "-"\n'
                '    Suffix = ".skf"\n'
                "  }\n"
                "  MaxAngularMomentum {\n"
            )
            for atom in atom_types:
                input += f'    {atom} = "{self.atom_dict[atom]}"\n'
            input += "  }\n"
            if mol.is_periodic:
                input += "  kPointsAndWeights = { 0.0 0.0 0.0 1.0 }\n"
            if self.thirdorder:
                input += "  ThirdOrderFull = Yes\n" "  HubbardDerivs {\n"
                for atom in atom_types:
                    input += f"    {atom} = {self.hubbard_derivs[atom]}\n"
                input += "  }\n" "  HCorrection = Damping {\n" "    Exponent = 4.00\n" "  }\n"
            if self.dispersion:
                input += (
                    "  Dispersion = SimpleDftD3 {\n"
                    "    a1 = 0.746\n"
                    "    a2 = 4.191\n"
                    "    s6 = 1.0\n"
                    "    s8 = 3.209\n"
                    "  }\n"
                )
            input += "}\n"

        elif self.method == "xTB":
            if self.solver:
                input += f"  Solver = {self.solver} {{}}\n"
            self.parameters = "gfn2"
            input += '  Method = "GFN2-xTB"\n'
            if mol.is_periodic:
                input += "  kPointsAndWeights = { 0.0 0.0 0.0 1.0 }\n"
            input += "}\n"

        input += "\n" "ParserOptions {\n" "  ParserVersion = 11\n" "}"

        with open("dftb_in.hsd", "w") as inp:
            inp.writelines(input)

        return

    def spe(
        self,
        mol: System,
        ncores: int = None,
        maxcore=None,
        inplace: bool = False,
        remove_tdir: bool = True,
    ):
        """Single point energy calculation.

        Parameters
        ----------
        mol : System object
            Input molecule to use in the calculation.
        ncores : int, optional
            number of cores, by default all available cores
        maxcore : dummy variable
            dummy variable used for compatibility with Orca calculations
        inplace : bool, optional
            updates info for the input molecule instead of outputting a new molecule object,
            by default False
        remove_tdir : bool, optional
            Temporary work directory will be removed, by default True

        Returns
        -------
        newmol : System object
            Output molecule containing the new energies.
        """

        if ncores is None:
            ncores = get_ncores()

        logger.info(f"{mol.name}, charge {mol.charge} spin {mol.spin} - {self.method} SPE")
        logger.debug(f"Running DFTB+ calculation on {ncores} cores")

        tdir = mkdtemp(
            prefix=mol.name + "_",
            suffix=f"_{self.__output_suffix}_spe",
            dir=os.getcwd(),
        )

        with sh.pushd(tdir):
            self.write_input(
                mol=mol,
                job_info={"type": "spe"},
            )

            if self.parallel == "mpi":
                os.environ["OMP_NUM_THREADS"] = "1"
                cmd = f"mpirun -np {ncores} {cfg.MPI_FLAGS} {self.__DFTBPATH} > output.out 2>> output.err"
                logger.debug(f"Running DFTB+ with command: {cmd}")
                os.system(cmd)

            elif self.parallel == "nompi":
                os.environ["OMP_NUM_THREADS"] = f"{ncores}"
                cmd = f"{self.__DFTBPATH} > output.out 2>> output.err"
                logger.debug(f"Running DFTB+ with command: {cmd}")
                os.system(cmd)

            if inplace is False:
                newmol = copy.deepcopy(mol)
                self.parse_output(newmol)

            else:
                self.parse_output(mol)

            process_output(mol, self.__output_suffix, "spe", charge=mol.charge, spin=mol.spin)
            if remove_tdir:
                shutil.rmtree(tdir)

            if inplace is False:
                return newmol

    def opt(
        self,
        mol: System,
        latticeopt: bool = False,
        ncores: int = None,
        maxcore=None,
        inplace: bool = False,
        remove_tdir: bool = True,
    ):
        """Geometry optimization.

        Parameters
        ----------
        mol : System object
            Input molecule to use in the calculation.
        latticeopt : bool, optional
            If True, also optimize PBC conditions. By default, False
        ncores : int, optional
            number of cores, by default all available cores
        maxcore : dummy variable
            dummy variable used for compatibility with Orca calculations
        inplace : bool, optional
            updates info for the input molecule instead of outputting a new molecule object,
            by default False
        remove_tdir : bool, optional
            Temporary work directory will be removed, by default True

        Returns
        -------
        newmol : System object
            Output molecule containing the new energies.
        """

        if ncores is None:
            ncores = get_ncores()

        logger.info(f"{mol.name}, charge {mol.charge} spin {mol.spin} - {self.method} OPT")
        logger.debug(f"Running DFTB+ calculation on {ncores} cores")

        tdir = mkdtemp(
            prefix=mol.name + "_",
            suffix=f"_{self.__output_suffix}_opt",
            dir=os.getcwd(),
        )

        with sh.pushd(tdir):
            self.write_input(
                mol=mol,
                job_info={
                    "type": "opt",
                    "latticeopt": latticeopt,
                },
            )

            if self.parallel == "mpi":
                os.environ["OMP_NUM_THREADS"] = "1"
                cmd = f"mpirun -np {ncores} {cfg.MPI_FLAGS} {self.__DFTBPATH} > output.out 2>> output.err"
                logger.debug(f"Running DFTB+ with command: {cmd}")
                os.system(cmd)

            elif self.parallel == "nompi":
                os.environ["OMP_NUM_THREADS"] = f"{ncores}"
                cmd = f"{self.__DFTBPATH} > output.out 2>> output.err"
                logger.debug(f"Running DFTB+ with command: {cmd}")
                os.system(cmd)

            if inplace is False:
                newmol = copy.deepcopy(mol)
                newmol.geometry.load_xyz("geo_end.xyz")
                newmol.geometry.level_of_theory_geometry = self.level_of_theory
                self.parse_output(newmol)

            else:
                mol.geometry.load_xyz("geo_end.xyz")
                mol.geometry.level_of_theory_geometry = self.level_of_theory
                self.parse_output(mol)

            process_output(mol, self.__output_suffix, "spe", charge=mol.charge, spin=mol.spin)
            if remove_tdir:
                shutil.rmtree(tdir)

            if inplace is False:
                return newmol

    def md_nvt(
        self,
        mol: System,
        steps: int,
        timestep: float = 1.0,
        temperature: float = 298.0,
        mdrestartfreq: int = 100,
        box_side: float = None,
        ncores: int = None,
        maxcore=None,
        inplace: bool = False,
        remove_tdir: bool = True,
        compress_traj: bool = True,
    ):
        """Molecular Dynamics simulation in the Canonical Ensemble (NVT).

        Parameters
        ----------
        mol : System object
            Input molecule to use in the calculation.
        steps : int
            Total steps of the simulation
        timestep : float, optional
            Time step (in fs) for the simulation.
        temperature : float, optional
            Temperature (in Kelvin) of the simulation
        mdrestartfreq : int, optional
            MD information is printed to md.out every mdrestartfreq steps, by default 100
        box_side : float, optional
            for periodic systems, defines the length (in Å) of the box side
        ncores : int, optional
            number of cores, by default all available cores
        maxcore : dummy variable
            dummy variable used for compatibility with Orca calculations
        inplace : bool, optional
            updates info for the input molecule instead of outputting a new molecule object,
            by default False
        remove_tdir : bool, optional
            Temporary work directory will be removed, by default True
        compress_traj : bool, optional
            if True, parses the geo.end and md.out files into a single, smaller file, which
            is then zipped in an archive.

        Returns
        -------
        Ensemble
            Ensemble containing the NVT MD trajectory data
        """

        if ncores is None:
            ncores = get_ncores()

        if box_side is None:
            box_side = mol.box_side

        logger.info(f"{mol.name}, charge {mol.charge} spin {mol.spin} - {self.method} NVT MD")
        logger.debug(f"Running DFTB+ calculation on {ncores} cores")

        tdir = mkdtemp(
            prefix=mol.name + "_",
            suffix=f"_{self.__output_suffix}_md_nvt",
            dir=os.getcwd(),
        )

        with sh.pushd(tdir):
            self.write_input(
                mol=mol,
                job_info={
                    "type": "md_nvt",
                    "timestep": timestep,
                    "temperature": temperature,
                    "steps": steps,
                    "mdrestartfreq": mdrestartfreq,
                },
            )

            if self.parallel == "mpi":
                os.environ["OMP_NUM_THREADS"] = "1"
                cmd = f"mpirun -np {ncores} {cfg.MPI_FLAGS} {self.__DFTBPATH} > output.out 2>> output.err"
                logger.debug(f"Running DFTB+ with command: {cmd}")
                os.system(cmd)

            elif self.parallel == "nompi":
                os.environ["OMP_NUM_THREADS"] = f"{ncores}"
                cmd = f"{self.__DFTBPATH} > output.out 2>> output.err"
                logger.debug(f"Running DFTB+ with command: {cmd}")
                os.system(cmd)

            ensemble = Ensemble(split_multixyz(mol=mol, file="geo_end.xyz"))

            import random, string

            suffix = "".join(random.choices(string.ascii_letters + string.digits, k=4))

            if compress_traj:
                compress_dftb_trajectory(f"{mol.name}_{mol.charge}_{mol.spin}")
                os.makedirs("../MD_trajectories", exist_ok=True)
                shutil.move(
                    f"{mol.name}_{mol.charge}_{mol.spin}.zip",
                    f"../MD_trajectories/{mol.name}_{mol.charge}_{mol.spin}.zip",
                )

            save_dftb_trajectory(f"{mol.name}_{mol.charge}_{mol.spin}_{suffix}")

            if mol.is_periodic:
                with open(f"../MD_data/{mol.name}_{mol.charge}_{mol.spin}_{suffix}.pbc", "w") as f:
                    f.write(f"{mol.box_side}")

            process_output(mol, self.__output_suffix, "md_nvt", mol.charge, mol.spin)
            if remove_tdir:
                shutil.rmtree(tdir)

        ### --> CURRENTLY NOT WORKING, REFACTORING NEEDED
        # trajectory = MDTrajectory(f"{mol.name}_{mol.charge}_{mol.spin}_{suffix}", self.method)
        ### <--

        return ensemble

    def simulated_annealing(
        self,
        mol: System,
        start_temp: float = 1.0,
        target_temp: float = 2000.0,
        ramp_steps: int = 500,
        hold_steps: int = 1000,
        timestep: float = 1.0,
        mdrestartfreq: int = 100,
        box_side: float = None,
        ncores: int = None,
        maxcore=None,
        inplace: bool = False,
        remove_tdir: bool = True,
        compress_traj: bool = True,
    ):
        """Molecular Dynamics simulated annealing simulation in the Canonical Ensemble (NVT)

        Parameters
        ----------
        mol : System object
            Input molecule to use in the calculation.
        start_temp: float, optional
            Starting temperature (default, 1K)
        target_temp: float, optional
            Maximum temperature reached during the simulation (default, 2000K)
        ramp_steps: int, optional
            Number of MD steps for the heating/cooling ramps (default, 500 steps)
        hold_steps: int, optional
            Number of MD steps held at target_temp (default, 1000 steps)
        timestep : float, optional
            Time step (in fs) for the simulation.
        mdrestartfreq : int, optional
            MD information is printed to md.out every mdrestartfreq steps, by default 100
        box_side : float, optional
            for periodic systems, defines the length (in Å) of the box side
        ncores : int, optional
            number of cores, by default all available cores
        maxcore : dummy variable
            dummy variable used for compatibility with Orca calculations
        inplace : bool, optional
            updates info for the input molecule instead of outputting a new molecule object,
            by default False
        remove_tdir : bool, optional
            Temporary work directory will be removed, by default True
        compress_traj : bool, optional
            if True, parses the geo.end and md.out files into a single, smaller file.

        Returns
        -------
        System
            System obtained at the end of the simulated annealing
        """

        if ncores is None:
            ncores = get_ncores()

        if box_side is None:
            box_side = mol.box_side

        logger.info(f"{mol.name}, charge {mol.charge} spin {mol.spin} - {self.method} Simulated Annealing")
        logger.debug(f"Running DFTB+ calculation on {ncores} cores")
        logger.debug(
            f"Heating/cooling between {start_temp}K and {target_temp}K for {ramp_steps} steps and holding max temp for {hold_steps} steps"
        )

        tdir = mkdtemp(
            prefix=mol.name + "_",
            suffix=f"_{self.__output_suffix}_anneal",
            dir=os.getcwd(),
        )

        with sh.pushd(tdir):
            self.write_input(
                mol=mol,
                job_info={
                    "type": "simulated_annealing",
                    "timestep": timestep,
                    "start_temp": start_temp,
                    "ramp_steps": ramp_steps,
                    "target_temp": target_temp,
                    "hold_steps": hold_steps,
                    "mdrestartfreq": mdrestartfreq,
                },
            )

            if self.parallel == "mpi":
                os.environ["OMP_NUM_THREADS"] = "1"
                cmd = f"mpirun -np {ncores} {cfg.MPI_FLAGS} {self.__DFTBPATH} > output.out 2>> output.err"
                logger.debug(f"Running DFTB+ with command: {cmd}")
                os.system(cmd)

            elif self.parallel == "nompi":
                os.environ["OMP_NUM_THREADS"] = f"{ncores}"
                cmd = f"{self.__DFTBPATH} > output.out 2>> output.err"
                logger.debug(f"Running DFTB+ with command: {cmd}")
                os.system(cmd)

            if inplace is False:
                newmol = copy.deepcopy(mol)
                newmol.geometry.load_xyz("geo_end.xyz")
                newmol.geometry.level_of_theory_geometry = self.level_of_theory

            else:
                mol.geometry.load_xyz("geo_end.xyz")
                mol.geometry.level_of_theory_geometry = self.level_of_theory

            import random, string

            suffix = "".join(random.choices(string.ascii_letters + string.digits, k=4))

            if compress_traj:
                compress_dftb_trajectory(mol.name)
                os.makedirs("../MD_trajectories", exist_ok=True)
                shutil.move(f"{mol.name}.zip", f"../MD_trajectories/{mol.name}.zip")

            save_dftb_trajectory(f"{mol.name}_{suffix}")

            if mol.is_periodic:
                with open(f"../MD_data/{mol.name}_{suffix}.pbc", "w") as f:
                    f.write(f"{mol.box_side}")

            process_output(mol, self.__output_suffix, "anneal", mol.charge, mol.spin)
            if remove_tdir:
                shutil.rmtree(tdir)

        ### --> CURRENTLY NOT WORKING, REFACTORING NEEDED
        # trajectory = MDTrajectory(f"{mol.name}_{mol.charge}_{mol.spin}_{suffix}", self.method)
        ### <--

        if inplace is False:
            return newmol

    def parse_output(self, mol: System) -> None:
        """
        The function will parse a DFTB+ output file automatically looking for all the
        relevant numerical properties derived form a calculation. All the properties of the
        given molecule will be set or updated.

        Parameters
        ----------
        mol: System
            The System to which the properties must be written to.

        Raises
        ------
        RuntimeError
            Exception raised if the given path to the output file is not valid.
        """

        if not os.path.isfile("output.out"):
            raise RuntimeError("Cannot parse output, the `output.out` file does not exist.")

        with open("output.out", "r") as outfile:
            for line in outfile:
                if "ERROR!" in line:
                    logger.error("Error occurred during DFTB+ calculation.")
                    raise RuntimeError("Error occurred during DFTB+ calculation")

        # Parse the final single point energy and the free energy correction
        # ----------------------------------------------------------------------------------
        with open("output.out", "r") as out:
            for line in out:
                if "Total Energy" in line:
                    electronic_energy = float(line.split()[2])
                    mol.properties.set_electronic_energy(electronic_energy, self)
