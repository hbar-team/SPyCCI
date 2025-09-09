import os, copy, sh, shutil
import numpy as np
from typing import Dict
from tempfile import mkdtemp
from spycci.config import get_ncores
from spycci.core.base import Engine
from spycci.systems import System
from spycci.tools import process_output, dissociation_check, cyclization_check
from spycci.tools.internaltools import clean_suffix
from spycci.core.dependency_finder import locate_xtb

import logging

logger = logging.getLogger(__name__)


class XtbInput(Engine):
    """
    Interface for running xTB calculations

    Parameters
    ----------
    method : str, optional
        level of theory, by default "gfn2"
    solvent : str, optional
        ALPB solvent, by default no solvent (vacuum)
    optionals : str, optional
        optional keywords/flags, by default ""
    XTBPATH: str, optional
        the path to the xtb executable. If set to None (default) the xtb executable will
        be loaded automatically.
    """

    def __init__(
        self,
        method: str = "gfn2",
        solvent: str = None,
        optionals: str = "",
        XTBPATH: str = None,
    ) -> None:
        super().__init__(method)

        self.solvent = solvent
        self.optionals = optionals
        self.__XTBPATH = locate_xtb(XTBPATH if XTBPATH else "xtb")

        self.level_of_theory += f" | solvent: {solvent}"
        self.__output_suffix = f"xtb_{self.method}_"
        self.__output_suffix += "vacuum" if solvent is None else f"{self.solvent}"
        self.__output_suffix = clean_suffix(self.__output_suffix)

    def write_input(self, mol: System, job_info: dict) -> None:
        input = f"$chrg {mol.charge}\n"
        input += f"$spin {mol.spin-1}\n"

        input += "$write\n"
        input += "   spin population=true\n"
        if job_info["save_cubes"]:
            input += "   density=true\n"
            input += "   spin density=true\n"
            input += "$cube\n"
            input += f"   step={job_info['cube_step']}\n"

        input += "$end\n"

        with open("input.inp", "w") as inp:
            inp.writelines(input)

    def spe(
        self,
        mol: System,
        ncores: int = None,
        maxcore=None,
        save_cubes: bool = False,
        cube_step: float = 0.1,
        inplace: bool = False,
        remove_tdir: bool = True,
    ):
        """Single point energy calculation.

        Parameters
        ----------
        mol : System object
            Input system to use in the calculation.
        ncores : int, optional
            number of cores, by default all available cores
        maxcore : dummy variable
            dummy variable used for compatibility with Orca calculations
        save_cubes: bool, optional
            if set to True, will save a cube file containing electronic and spin densities,
            by default False.
        cube_step: int, optional
            grid spacing for cube files, in Bohrs (default 0.4)
        inplace : bool, optional
            updates info for the input system instead of outputting a new system object,
            by default False
        remove_tdir : bool, optional
            Temporary work directory will be removed, by default True

        Returns
        -------
        newmol : System object
            Output system containing the new energies.
        """

        if ncores is None:
            ncores = get_ncores()

        logger.info(f"{mol.name}, charge {mol.charge} spin {mol.spin} - {self.method} SPE")
        logger.debug(f"Running xTB calculation on {ncores} cores")

        tdir = mkdtemp(
            prefix=mol.name + "_",
            suffix=f"_{self.__output_suffix}_spe",
            dir=os.getcwd(),
        )

        with sh.pushd(tdir):
            self.write_input(
                mol,
                job_info={
                    "save_cubes": save_cubes,
                    "cube_step": cube_step,
                },
            )

            mol.geometry.write_xyz(f"{mol.name}.xyz")

            if self.solvent:
                cmd = f"{self.__XTBPATH} --input input.inp {mol.name}.xyz --{self.method} --alpb {self.solvent} -P {ncores} {self.optionals} > output.out 2>> output.err"
                logger.debug(f"Running xTB with command: {cmd}")
                os.system(cmd)

            else:
                cmd = f"{self.__XTBPATH} --input input.inp {mol.name}.xyz --{self.method} -P {ncores} {self.optionals} > output.out 2>> output.err"
                logger.debug(f"Running xTB with command: {cmd}")
                os.system(cmd)

            if inplace is False:
                newmol = System(f"{mol.name}.xyz", charge=mol.charge, spin=mol.spin)
                newmol.properties = copy.copy(mol.properties)
                self.parse_output(newmol)

            else:
                self.parse_output(mol)

            process_output(mol, self.__output_suffix, "spe", mol.charge, mol.spin, save_cubes=save_cubes)

            if remove_tdir:
                shutil.rmtree(tdir)

        if inplace is False:
            return newmol

    def opt(
        self,
        mol: System,
        ncores: int = None,
        maxcore=None,
        save_cubes: bool = False,
        cube_step: float = 0.1,
        inplace: bool = False,
        remove_tdir: bool = True,
    ):
        """Geometry optimization + frequency analysis.

        Parameters
        ----------
        mol : System object
            Input system to use in the calculation
        ncores : int, optional
            number of cores, by default all available cores
        maxcore : dummy variable
            dummy variable used for compatibility with Orca calculations
        save_cubes: bool, optional
            if set to True, will save a cube file containing electronic and spin densities,
            by default False.
        cube_step: int, optional
            grid spacing for cube files, in Bohrs (default 0.4)
        inplace : bool, optional
            updates info for the input system instead of outputting a new system object,
            by default False
        remove_tdir : bool, optional
            Temporary work directory will be removed, by default True

        Returns
        -------
        newmol : System object
            Output system containing the new geometry and energies.

        If a dissociation or a cyclization is observed, ignore the calculation and return the
        original "mol" system.
        """

        if ncores is None:
            ncores = get_ncores()

        logger.info(f"{mol.name}, charge {mol.charge} spin {mol.spin} - {self.method} OPT")
        logger.debug(f"Running xTB calculation on {ncores} cores")

        tdir = mkdtemp(
            prefix=mol.name + "_",
            suffix=f"_{self.__output_suffix}_opt",
            dir=os.getcwd(),
        )

        with sh.pushd(tdir):
            self.write_input(
                mol,
                job_info={
                    "save_cubes": save_cubes,
                    "cube_step": cube_step,
                },
            )

            mol.geometry.write_xyz(f"{mol.name}.xyz")

            if self.solvent:
                cmd = f"{self.__XTBPATH} --input input.inp {mol.name}.xyz --{self.method} --alpb {self.solvent} --ohess -P {ncores} {self.optionals} > output.out 2>> output.err"
                logger.debug(f"Running xTB with command: {cmd}")
                os.system(cmd)

            else:
                cmd = f"{self.__XTBPATH} --input input.inp {mol.name}.xyz --{self.method} --ohess -P {ncores} {self.optionals} > output.out 2>> output.err"
                logger.debug(f"Running xTB with command: {cmd}")
                os.system(cmd)

            if dissociation_check() is True:
                logger.error(f"Dissociation spotted for {mol.name}.")
                mol.flags.append(f"Dissociation occurred during geometry optimization with {self.method}.")
                return None
            elif cyclization_check(f"{mol.name}.xyz", "xtbopt.xyz") is True:
                logger.error(f"Cyclization change spotted for {mol.name}.")
                mol.flags.append(f"Cyclization change occurred during geometry optimization with {self.method}.")
                return None
            else:
                if inplace is False:
                    newmol = System(f"{mol.name}.xyz", charge=mol.charge, spin=mol.spin)
                    newmol.geometry.load_xyz("xtbopt.xyz")
                    newmol.geometry.level_of_theory_geometry = self.level_of_theory

                    self.parse_output(newmol)

                else:
                    mol.geometry.load_xyz("xtbopt.xyz")
                    mol.geometry.level_of_theory_geometry = self.level_of_theory

                    self.parse_output(mol)

                process_output(mol, self.__output_suffix, "opt", mol.charge, mol.spin, save_cubes=save_cubes)

                if remove_tdir:
                    shutil.rmtree(tdir)

        if inplace is False:
            return newmol

    def freq(
        self,
        mol: System,
        ncores: int = None,
        maxcore=None,
        save_cubes: bool = False,
        cube_step: float = 0.1,
        inplace: bool = False,
        remove_tdir: bool = True,
    ):
        """Frequency analysis.

        Parameters
        ----------
        mol : System object
            input system to use in the calculation
        ncores : int, optional
            number of cores, by default all available cores
        maxcore : dummy variable
            dummy variable used for compatibility with Orca calculations
        save_cubes: bool, optional
            if set to True, will save a cube file containing electronic and spin densities,
            by default False.
        cube_step: int, optional
            grid spacing for cube files, in Bohrs (default 0.4)
        inplace : bool, optional
            updates info for the input system instead of outputting a new system object,
            by default False
        remove_tdir : bool, optional
            temporary work directory will be removed, by default True

        Returns
        -------
        newmol : System object
            Output system containing the new energies.
        """

        if ncores is None:
            ncores = get_ncores()

        logger.info(f"{mol.name}, charge {mol.charge} spin {mol.spin} - {self.method} FREQ")
        logger.debug(f"Running xTB calculation on {ncores} cores")

        tdir = mkdtemp(
            prefix=mol.name + "_",
            suffix=f"_{self.__output_suffix}_freq",
            dir=os.getcwd(),
        )

        with sh.pushd(tdir):
            self.write_input(
                mol,
                job_info={
                    "save_cubes": save_cubes,
                    "cube_step": cube_step,
                },
            )

            mol.geometry.write_xyz(f"{mol.name}.xyz")

            if self.solvent:
                cmd = f"{self.__XTBPATH} --input input.inp {mol.name}.xyz --{self.method} --alpb {self.solvent} --hess -P {ncores} {self.optionals} > output.out 2>> output.err"
                logger.debug(f"Running xTB with command: {cmd}")
                os.system(cmd)

            else:
                cmd = f"{self.__XTBPATH} --input input.inp {mol.name}.xyz --{self.method} --hess -P {ncores} {self.optionals} > output.out 2>> output.err"
                logger.debug(f"Running xTB with command: {cmd}")
                os.system(cmd)

            if inplace is False:
                newmol = System(f"{mol.name}.xyz", charge=mol.charge, spin=mol.spin)

                newmol.properties = copy.copy(mol.properties)

                self.parse_output(newmol)

            else:
                self.parse_output(mol)

            process_output(mol, self.__output_suffix, "freq", mol.charge, mol.spin, save_cubes=save_cubes)

            if remove_tdir:
                shutil.rmtree(tdir)

        if inplace is False:
            return newmol

    def parse_output(self, mol: System) -> None:
        """
        The function will parse an xTB output file automatically looking for all the
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
                if "[ERROR] Program stopped due to fatal error" in line:
                    logger.error("Error occurred during xTB calculation.")
                    raise RuntimeError("Error occurred during xTB calculation")

        # Parse the final single point energy and the Gibbs free energy correction
        # ----------------------------------------------------------------------------------
        with open("output.out", "r") as out:
            for line in out:
                if "TOTAL ENERGY" in line:
                    electronic_energy = float(line.split()[-3])
                    mol.properties.set_electronic_energy(electronic_energy, self)
                if "G(RRHO) contrib." in line:
                    free_energy_correction = float(line.split()[-3])
                    mol.properties.set_free_energy_correction(free_energy_correction, self)
                if "TOTAL FREE ENERGY" in line:
                    gibbs_free_energy = float(line.split()[-3])
                    if not np.isclose(gibbs_free_energy, mol.properties.gibbs_free_energy, rtol=1e-9):
                        raise RuntimeError("Computed Gibbs Free Energy differs from parsed one. This is unexpected.")

        # Parse the Mulliken atomic charges and spin populations
        # ----------------------------------------------------------------------------------
        mulliken_charges, mulliken_spins = [], []
        spin_available = False
        with open("output.out", "r") as file:
            for line in file:
                # read the file until the charges section is reached
                if "#   Z          covCN" in line:
                    # Iterate over the whole section reading line by line
                    while True:
                        buffer = file.readline()
                        if len(buffer.split()) == 0:
                            break
                        else:
                            data = buffer.split()
                            mulliken_charges.append(float(data[4]))

                # Check if the section contains also the "(R)spin-density population" line
                if "Mulliken population" in line:
                    spin_available = True

                    # Iterate over the whole section reading line by line
                    while True:
                        buffer = file.readline()
                        if len(buffer.split()) == 0:
                            break
                        else:
                            data = buffer.split()
                            mulliken_spins.append(float(data[1]))

            if not spin_available:
                mulliken_spins = [0.0] * len(mulliken_charges)

        if mulliken_charges != []:
            mol.properties.set_mulliken_charges(mulliken_charges, self)
            mol.properties.set_mulliken_spin_populations(mulliken_spins, self)

    @property
    def output_suffix(self) -> str:
        """
        Suffix used to compose the name of calculation output files

        Returns
        -------
        str
            The output suffix string
        """
        return self.__output_suffix
