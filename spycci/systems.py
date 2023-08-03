from __future__ import annotations

import os, json
import numpy as np
import logging

from typing import List
from enum import Enum
from copy import deepcopy

from spycci.constants import kB
from spycci.config import __JSON_VERSION__
from spycci.core.geometry import MolecularGeometry
from spycci.core.properties import Properties

logger = logging.getLogger(__name__)


class System:
    """
    The System object describes a generic molecular system described at a given level of
    theory. A system is defined based on a molecular geometry, a charge, a spin multiplicity
    and, one a level of theory is selected, by a set of computed properties.

    Parameters
    ----------
    filepath : str
        The path to the file containing the system geometry or data
    charge : int, optional
        The total charge of the system. (Default: 0 neutral)
    spin : int, optional
        The total spin multiplicity of the system. (Default: 1 singlet)
    box_side : float, optional
        For periodic systems, defines the length (in Å) of the box side

    Raises
    ------
    ValueError
        Exception raised when the `.xyz` file path is invalid
    """

    def __init__(
        self,
        filepath: str,
        charge: int = 0,
        spin: int = 1,
        box_side: float = None,
    ) -> None:

        if not os.path.isfile(filepath):
            raise ValueError(f"The specified file `{filepath}` does not exist.")

        if filepath.endswith(".xyz"):
            self.name = os.path.basename(filepath).strip(".xyz")
            self.__charge: int = charge
            self.__spin: int = spin
            self.__box_side = box_side
            self.__geometry: MolecularGeometry = MolecularGeometry.from_xyz(filepath)
            self.properties: Properties = Properties()
            self.flags: list = []

        elif filepath.endswith(".json"):

            with open(filepath, "r") as jsonfile:
                data = json.load(jsonfile)

            data = json_parser(data)

            self.name = data["Name"]
            self.__charge = data["Charge"]
            self.__spin = data["Spin"]
            self.__box_side = data["Box Side"]
            self.__geometry = MolecularGeometry().from_dict(data["Geometry"])
            self.properties = Properties().from_dict(data["Properties"])
            self.flags = data["Flags"]

        else:
            raise RuntimeError("The specified format is not supported")

    def save_json(self, path: str) -> None:
        """
        Saves a JSON representation of the object that can be stored on disk and loaded (using
        the class constructor with the option SupportedTypes.JSON) at a later time to
        re-generate ad identical System object.

        Arguments
        ---------
        path: str
            The path to the .json file that must be created.
        """
        data = {}
        data["__JSON_VERSION__"] = __JSON_VERSION__
        data["Name"] = self.name
        data["Charge"] = self.__charge
        data["Spin"] = self.__spin
        data["Box Side"] = self.__box_side
        data["Geometry"] = self.__geometry.to_dict()
        data["Properties"] = self.properties.to_dict()
        data["Flags"] = self.flags

        with open(path, "w") as jsonfile:
            json.dump(data, jsonfile)

    @property
    def geometry(self) -> MolecularGeometry:
        return self.__geometry

    @geometry.setter
    def geometry(self, new_geometry: MolecularGeometry) -> None:

        if type(new_geometry) != MolecularGeometry:
            raise TypeError("The geometry attribute must be of type MolecularGeometry")

        elif new_geometry.atomcount == 0:
            raise ValueError("The geometry object cannot be empty or not initialized")

        self.__geometry = new_geometry
        logger.info(f"Geometry changed: clearing properties for {self.name}")
        self.properties = Properties()

    @property
    def charge(self) -> int:
        return self.__charge

    @charge.setter
    def charge(self, new_charge: int) -> None:

        if type(new_charge) != int:
            logger.error(f"Charge {new_charge} is invalid. Must be an integer.")
            raise TypeError("Charge must be an integer value")
        
        self.__charge = new_charge
        logger.info(f"Charge changed: clearing properties for {self.name}")
        self.properties = Properties()

    @property
    def spin(self) -> int:
        return self.__spin

    @spin.setter
    def spin(self, new_spin: int) -> None:

        if type(new_spin) != int or new_spin<1:
            logger.error(f"Spin multiplicity {new_spin} is invalid. Must be a positive integer.")
            raise TypeError("Spin multiplicity must be an integer value")
        
        self.__spin = new_spin
        logger.info(f"Spin changed: clearing properties for {self.name}")
        self.properties = Properties()

    @property
    def box_side(self) -> float:
        return self.__box_side

    @box_side.setter
    def box_side(self, value: float) -> None:
        self.__box_side = value
        logger.info(f"Box side changed: clearing properties for {self.name}")
        self.properties = Properties()

    def __str__(self):
        info = "=========================================================\n"
        info += f"SYSTEM: {self.name}\n"
        info += "=========================================================\n\n"
        info += f"Number of atoms: {self.geometry.atomcount}\n"
        info += f"Charge: {self.charge}\n"
        info += f"Spin multiplicity: {self.spin}\n"

        if self.box_side:
            info += f"Periodic system with box size: {self.box_side:.4f} Å\n"
        info += "\n"

        info += "********************** GEOMETRY *************************\n\n"
        info += f"Total system mass: {self.geometry.mass:.4f} amu\n\n"

        info += "----------------------------------------------\n"
        info += " index  atom    x (Å)      y (Å)      z (Å)   \n"
        info += "----------------------------------------------\n"
        for idx, (atom, coordinates) in enumerate(self.geometry):
            info += f" {idx:<6}{atom:^6}"
            for c in coordinates:
                info += "{0:^11}".format(f"{c:.5f}")
            info += "\n"
        info += "----------------------------------------------\n\n"

        info += "********************** PROPERTIES *************************\n\n"
        info += f"Geometry level of theory: {self.geometry.level_of_theory_geometry}\n"
        info += (
            f"Electronic level of theory: {self.properties.level_of_theory_electronic}\n"
        )
        info += f"Vibronic level of theory: {self.properties.level_of_theory_vibronic}\n\n"
        info += f"Electronic energy: {self.properties.electronic_energy} Eh\n"
        info += f"Vibronic energy: {self.properties.vibronic_energy} Eh\n"
        info += f"Helmholtz free energy: {self.properties.helmholtz_free_energy} Eh\n"
        info += f"Gibbs free energy: {self.properties.gibbs_free_energy} Eh\n"
        info += f"pKa: {self.properties.pka}\n\n"

        if self.properties.mulliken_charges != []:

            info += f"MULLIKEN ANALYSIS\n"
            info += "----------------------------------------------\n"
            info += " index  atom   charge    spin\n"
            info += "----------------------------------------------\n"
            for idx, (atom, charge, spin) in enumerate(
                zip(
                    self.geometry.atoms,
                    self.properties.mulliken_charges,
                    self.properties.mulliken_spin_populations,
                )
            ):
                info += f" {idx:<6}{atom:^6}"
                info += "{0:^10}{1:^10}\n".format(
                    f"{charge:.5f}",
                    f"{spin:.5f}",
                )
            info += "\n"

        if self.properties.condensed_fukui_mulliken != {}:

            info += f"CONDENSED FUKUI - MULLIKEN\n"
            info += "----------------------------------------------\n"
            info += " index  atom    f+      f-      f0\n"
            info += "----------------------------------------------\n"
            for idx, (atom, fplus, fminus, fzero) in enumerate(
                zip(
                    self.geometry.atoms,
                    self.properties.condensed_fukui_mulliken["f+"],
                    self.properties.condensed_fukui_mulliken["f-"],
                    self.properties.condensed_fukui_mulliken["f0"],
                )
            ):
                info += f" {idx:<6}{atom:^6}"
                info += "{0:^10}{1:^10}{2:^10}\n".format(
                    f"{fplus:.5f}",
                    f"{fminus:.5f}",
                    f"{fzero:.5f}",
                )
            info += "\n"

        if self.properties.hirshfeld_charges != []:

            info += f"HIRSHFELD ANALYSIS\n"
            info += "----------------------------------------------\n"
            info += " index  atom   charge    spin\n"
            info += "----------------------------------------------\n"
            for idx, (atom, charge, spin) in enumerate(
                zip(
                    self.geometry.atoms,
                    self.properties.hirshfeld_charges,
                    self.properties.hirshfeld_spin_populations,
                )
            ):
                info += f" {idx:<6}{atom:^6}"
                info += "{0:^10}{1:^10}\n".format(
                    f"{charge:.5f}",
                    f"{spin:.5f}",
                )
            info += "\n"

        if self.properties.condensed_fukui_hirshfeld != {}:

            info += f"CONDENSED FUKUI - HIRSHFELD\n"
            info += "----------------------------------------------\n"
            info += " index  atom    f+      f-      f0\n"
            info += "----------------------------------------------\n"
            for idx, (atom, fplus, fminus, fzero) in enumerate(
                zip(
                    self.geometry.atoms,
                    self.properties.condensed_fukui_hirshfeld["f+"],
                    self.properties.condensed_fukui_hirshfeld["f-"],
                    self.properties.condensed_fukui_hirshfeld["f0"],
                )
            ):
                info += f" {idx:<6}{atom:^6}"
                info += "{0:^10}{1:^10}{2:^10}\n".format(
                    f"{fplus:.5f}",
                    f"{fminus:.5f}",
                    f"{fzero:.5f}",
                )
            info += "\n"

        if self.flags != []:
            info += "********************** WARNINGS **************************\n\n"
            for warning in self.flags:
                info += f"{warning}\n"

        return info

    def write_gen(self, gen_file: str, box_side: float = None):
        """
        Writes the current geometry to a `.gen` file.

        Parameters
        ----------
        gen_file : str
            path to the output `.gen` file
        box_side : float, optional
            for periodic systems, defines the length (in Å) of the box side
        """

        if box_side is None:
            box_side = self.box_side

        with open(gen_file, "w") as file:

            file.write(f" {str(self.geometry.atomcount)} ")
            file.write("S\n" if self.is_periodic else "C\n")

            atom_types = []
            for element in self.geometry.atoms:
                if element not in atom_types:
                    atom_types.append(element)

            for atom in atom_types:
                file.write(f" {atom}")
            file.write("\n")

            i = 1
            for atom, coordinates in self.geometry:
                line = (
                    f"{atom}\t"
                    + f"{coordinates[0]}\t"
                    + f"{coordinates[1]}\t"
                    + f"{coordinates[2]}\t"
                    + "\n"
                )
                for index, atom_type in enumerate(atom_types):
                    if line.split()[0] == atom_type:
                        file.write(f"{i} {line.replace(atom_type, str(index + 1))}")
                        i += 1

            if self.is_periodic:
                file.write(f" 0.000 0.000 0.000\n")
                file.write(f" {box_side} 0.000 0.000\n")
                file.write(f" 0.000 {box_side} 0.000\n")
                file.write(f" 0.000 0.000 {box_side}")

    @property
    def is_periodic(self) -> bool:
        """
        Indicates if the system is periodic or not

        Returns
        -------
        bool
            True if the system is periodic (the `box_side` is not None), False otherwise.
        """
        return True if self.box_side is not None else False



def json_parser(input: dict) -> dict:
    """
    The parser takes an input dictionary from an arbitrary .json file version and converts it into a dictionary
    compatible with the latest .json file standard.

    Arguments
    ---------
    input: dict
        The dictionary containing the data that used to be required to fully describe a System object in a previous
        version of the internal .json standard
    
    Returns
    -------
    dict
        The dictionary containing the data required to fully describe a System object in the new .json standard
    """

    output = deepcopy(input)

    if "__JSON_VERSION__" not in input or input["__JSON_VERSION__"] < 1:
        
        logger.warning("Detected JSON version <1: Updating dictionary content to version 1")

        # In version 1 of the JSON format the vibrational data field has been added
        output["Properties"]["Vibrational data"] = None
        output["__JSON_VERSION__"] = 1
    
    output["__JSON_VERSION__"] = __JSON_VERSION__

    return output



class Ensemble:
    """
    Ensemble object, containing a series of System objects.

    Parameters
    ----------
    systems : List[System]
        The list of System objects to be included in the Ensemble.

    Attributes
    ----------
    name : str
        Name of the system represented in the ensemble, taken from the first element of
        the ensemble
    systems : List[System]
        The list of System objects in the Ensemble.
    properties : PropertiesArchive
        The property archive containing the average ensamble properties calculated at
        various levels of theory.
    """

    def __init__(self, systems: List[System]) -> None:

        if len(systems) == 0:
            raise ValueError("Cannot operate on an empty systems array")

        if any(system.geometry.atoms != systems[0].geometry.atoms for system in systems):
            raise RuntimeError("Different systems encountered in list")

        self.name: str = systems[0].name
        self.systems: List[System] = systems
        self.helmholtz_free_energy: float = None

    def __iter__(self) -> System:
        for item in self.systems:
            yield item

    def __getitem__(self, index: int) -> System:
        if index < 0 or index >= len(self.systems):
            raise ValueError("Index out of bounds")

        return self.systems[index]

    def __len__(self) -> int:
        return len(self.systems)

    @property
    def atomcount(self) -> int:
        """
        The number of atoms in the system

        Returns
        -------
        int
            The total number of atoms
        """
        return self.systems[0].geometry.atomcount

    def add(self, systems: List[System]):
        """
        Append more Systems to the ensemble

        Parameters
        ----------
        systems : List[System]
            The list of systems to be added to the ensamble
        """
        if any(system.geometry.atoms != systems[0].geometry.atoms for system in systems):
            raise RuntimeError("Different systems encountered in list")

        for system in systems:
            self.systems.append(system)

    def boltzmann_average(
        self,
        temperature: float = 297.15,
    ) -> float:
        """
        Calculates the average free Helmholtz energy of the ensemble (in Hartree), weighted
        for each molecule by its Boltzmann factor.

        Parameters
        ----------
        temperature : float
            temperature at which to calculate the Boltzmann average, by default 297.15 K

        Returns
        -------
        float
            The total Helmholtz free energy of the ensemble.

        NOTE: the vibronic contributions are included in the electronic component, which
        actually contains the TOTAL energy of the system. Maybe in the future I'll think of
        how to separate the two contributions - LB
        """
        energies = []

        for system in self.systems:
            if system.properties.vibronic_energy is None:
                energies.append(system.properties.electronic_energy)
            else:
                energies.append(
                    system.properties.electronic_energy + system.properties.vibronic_energy
                )

        # Compute the relative energy of each system in respect to the minimum to avoid overflows
        # when computing exponential of large magnitude values
        dE = [energy - min(energies) for energy in energies]

        # Compute the relative partition function starting from the relative energy list
        relative_Z = np.sum(np.exp([-energy / (kB * temperature) for energy in dE]))

        # Compute the populations for each system given the boltzmann distribution
        populations = [np.exp(-energy / (kB * temperature)) / relative_Z for energy in dE]

        # Compute the weighted energy values by including the population of each state
        weighted_energies = [
            energy * population for energy, population in zip(energies, populations)
        ]

        # Compute the entropy of the system
        boltzmann_entropy = -kB * np.sum(populations * np.log(populations))

        # Compute the helmotz free energy for the ensamble
        self.helmholtz_free_energy = (
            np.sum([weighted_energies]) - temperature * boltzmann_entropy
        )


# class MDTrajectory:
#     """
#     Iterator class for MD trajectories. Data is computed only when accessing the
#     elements of the object (via __getitem__ or __iter__)

#     Parameters
#     ----------
#     traj_path : str
#         path (prefix) of the trajectory files used for creating the MD run
#     method : str
#         level of theory at which the simulation was ran
#     """
#     def __init__(self, traj_filepath: str, method: str) -> None:

#         self.name = traj_filepath
#         self.method = method

#         self.md_out = f"MD_data/{traj_filepath}_md.out"
#         self.geo_end = f"MD_data/{traj_filepath}_geo_end.xyz"

#         self.box_side = None
#         if os.path.exists(f"MD_data/{traj_filepath}.pbc"):
#             with open(f"MD_data/{traj_filepath}.pbc") as f:
#                 self.box_side = float(f.read())

#         with open(self.geo_end, "r") as f:

#             self.atomcount = int(f.readline())
#             first_iter = int(f.readline().split()[-1])

#             for line in f:

#                 if "iter" in line:
#                     second_iter = int(line.split()[-1])
#                     self.mdrestartfreq = second_iter - first_iter
#                     break

#             for line in reversed(list(f)):
#                 if "iter" in line:
#                     nframes = int(line.split()[-1])
#                     break

#         self.frames = list(range(0, (nframes // self.mdrestartfreq) + 1))

#     def __iter__(self):
#         for index in self.frames:
#             with open(self.geo_end, "r") as geo_end:
#                 yield self.__getitem__(index, geo_end)

#     def __getitem__(self, index, geo_end: TextIOWrapper = None):
#         """returns the System object corresponding to the requested index/MD step

#         Parameters
#         ----------
#         index : int
#             MD step in the simulation (as list index, not ACTUAL MD iter number)
#         geo_end : TextIOWrapper, optional
#             if __getitem__ is called from __iter__, allows passing the geo_end.xyz file so
#             it is open only once

#         Returns
#         -------
#         system
#             System object corresponding to the requested frame in the MD simulation
#         """

#         MDindex = self.frames[index] * self.mdrestartfreq
#         close_flag = False

#         if not geo_end:
#             close_flag = True
#             geo_end = open(self.geo_end, "r")

#         with open(f"{self.name}_{MDindex}.xyz", "w+") as out:

#             self.atomcount = int(geo_end.readline())
#             out.write(f"{self.atomcount}\n")
#             start = None

#             for i, line in enumerate(geo_end):

#                 if line == f"MD iter: {MDindex}\n":
#                     start = i
#                     out.write(line)

#                 if start and i > start and i < start + self.atomcount + 1:
#                     out.write(
#                         f"{line.split()[0]}\t{line.split()[1]}\t{line.split()[2]}\t{line.split()[3]}\n"
#                     )

#                 if start and i > start + self.atomcount + 1:
#                     break

#         if close_flag:
#             geo_end.close()

#         # !!! implement bytestream input for System !!!
#         system = System(f"{self.name}_{MDindex}.xyz", box_side=self.box_side)
#         os.remove(f"{self.name}_{MDindex}.xyz")

#         with open(self.md_out, "r") as md_out:
#             found = False
#             for line in md_out:
#                 if line == f"MD step: {MDindex}\n":
#                     found = True
#                 if "Total MD Energy" in line and found:
#                     system.energies[self.method] = Energies(
#                         method=self.method,
#                         electronic=line.split()[3],
#                         vibronic=None,
#                     )
#                     break

#         return system

#     def __len__(self):
#         return len(self.frames)
