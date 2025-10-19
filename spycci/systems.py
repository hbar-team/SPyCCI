from __future__ import annotations

import os, json
import numpy as np
import logging

from typing import List, Generator, Optional, Union
from copy import deepcopy

from spycci.constants import kB
from spycci.config import __JSON_VERSION__
from spycci.core.base import Engine
from spycci.core.geometry import MolecularGeometry
from spycci.core.properties import Properties

logger = logging.getLogger(__name__)


class System:
    """
    The System object describes a generic molecular system described at a given level of
    theory. A system is defined based on a molecular geometry, a charge, a spin multiplicity
    and, once a level of theory is selected, by a set of computed properties.

    Arguments
    ----------
    name : str
        The name of the system.
    geometry : MolecularGeometry
        The `MolecularGeometry` object from which the `System` can be constructred.
    charge : Optional[int]
        The total charge of the system. (Default: 0 neutral)
    spin : Optional[int]
        The total spin multiplicity of the system. (Default: 1 singlet)
    box_side : Optional[float]
        For periodic systems, defines the length (in Å) of the box side.

    Raises
    ------
    TypeError
        Exception raised if the geometry argument is not of the `MolecularGeometry` type.
    """
    def __init__(
        self,
        name: str,
        geometry: MolecularGeometry,
        charge: int = 0,
        spin: int = 1,
        box_side: Optional[float] = None,
    ) -> None:
        if type(geometry) != MolecularGeometry:
                raise TypeError("The `geometry` argument must be of type `MolecularGeometry`.")
            
        self.name = str(name)

        self.__geometry: MolecularGeometry = deepcopy(geometry)
        self.__geometry._MolecularGeometry__add_system_reset(self.__on_geometry_change)    # Set listener in MolecularGeometry class using mangled name

        self.__charge: int = charge
        self.__spin: int = spin
        self.__box_side = box_side

        self.__properties: Properties = Properties()
        self.__properties._Properties__add_check_geometry_level_of_theory(self.__check_geometry_level_of_theory)   # Set listener in Properties class using mangled name

        self.flags: list = []
        logger.debug(f"CREATED: System object {self.name} at ID: {hex(id(self))}.")
    
    def __on_geometry_change(self) -> None:
        """
        Function used by the `MolecularGeometry` listener to clear properties when molecular geometry has been changed.
        """
        logger.debug(f"CLEARED: Properties of {self.name} system (ID: {hex(id(self))}) due to molecular geometry change.")
        self.properties = Properties()
    
    def __check_geometry_level_of_theory(self, level_of_theory: str) -> None:
        """
        Function used by the `Properties` class listener to check compatibility of a newly provided
        level of theory with the currently adopted geometric level of theory. If the level of theory
        provided is different exception is raised. If the geometry level of theory is `None` the check
        is skipped silently.

        Argument
        --------
        level_of_theory: str
            The level of theory to be checked agains the geometry level of theory.
        
        Raises
        ------
        RuntimeError
            Exception raised if the proposed level of theory is different from the one set
            as the geometry level of theory.
        """
        if self.geometry.level_of_theory_geometry is not None:
            if level_of_theory != self.geometry.level_of_theory_geometry:
                raise RuntimeError("Mismatch between the user-provided level of theory and the one used to set geometry")
        
    @classmethod
    def from_xyz(
        cls, 
        path: str, 
        charge: int = 0,
        spin: int = 1,
        box_side: Optional[float] = None,
    ) -> System:
        """
        Construct a `System` object from the geometry encoded in a `.xyz` file.

        Arguments
        ----------
        path : str
            The path of the `.xyz` file.
        charge : Optional[int]
            The total charge of the system. (Default: 0 neutral)
        spin : Optional[int]
            The total spin multiplicity of the system. (Default: 1 singlet)
        box_side : Optional[float]
            For periodic systems, defines the length (in Å) of the box side.

        Raises
        ------
        FileNotFoundError
            Exception raised if the specified `.xyz` file cannot be found.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The specified XYZ file `{path}` does not exist.")

        name = os.path.basename(path).strip(".xyz")
        geometry = MolecularGeometry.from_xyz(path)

        obj = System(name, geometry, charge=charge, spin=spin, box_side=box_side)
        return obj
        
    @classmethod
    def from_json(cls, path: str) -> System:
        """
        Construct a `System` object from the data encoded in a SPyCCI `.json` file.

        Arguments
        ----------
        path : str
            The path of the `.json` file.
        
        Raises
        ------
        FileNotFoundError
            Exception raised if the specified `.json` file cannot be found.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The specified JSON file `{path}` does not exist.")

        with open(path, "r") as jsonfile:
            data = json.load(jsonfile)

        data = json_parser(data)

        name = data["Name"]
        geometry = MolecularGeometry().from_dict(data["Geometry"])
        charge = data["Charge"]
        spin = data["Spin"]
        box_side = data["Box Side"]

        obj = System(name, geometry, charge=charge, spin=spin, box_side=box_side)
        obj.properties = Properties().from_dict(data["Properties"])
        obj.flags = data["Flags"]

        return obj


    @classmethod
    def from_smiles(
        cls,
        name: str,
        smiles: str, 
        charge: int = 0,
        spin: int = 1,
        box_side: Optional[float] = None,
        force_uff: bool = False,
        default_mmffvariant: str = "MMFF94",
        use_small_ring_torsions: bool = True,
        use_macrocycle_torsions: bool = True,
        maxiter: int = 500,
        random_seed: int = -1,
    ) -> System:
        """
        The function returns a fully initialized `System` object from the given SMILES string.
        The geometry of the molecule is generated in 3D using the ETKDGv3 algorithm, followed by force field
        optimization using either MMFF or UFF, depending on availability and user preference.
        The resulting system includes molecular geometry, total charge, spin multiplicity, and optional
        periodic boundary conditions. All operations are carried out using the tools defined in the RDKit package.

        Arguments
        ---------
        name: str
            The name assigned to the molecular system.
        smiles: str
            A valid SMILES string representing the molecular structure to be converted into a full system.
        charge: int
            The net formal charge of the system. (Default: 0)
        spin: int
            The total spin multiplicity of the system. (Default: 1 singlet)
        box_side: Optional[float]
            The length of the side of the simulation box (in Å) used to define periodic boundary conditions.
            If set to `None`, the system is treated as non-periodic. (default: None)
        force_uff: bool
            If True, the geometry optimization will use the UFF force field even when MMFF parameters are available.
            This can be useful when consistency across a dataset using UFF is desired. (default: False)
        default_mmffvariant: str
            Specifies the MMFF variant to use when MMFF is selected. Valid options are "MMFF94" or "MMFF94s".
            "MMFF94s" is generally recommended for more accurate static geometries. (default: "MMFF94")
        use_small_ring_torsions: bool
            If True, enables special torsional sampling for small rings during 3D embedding.
            Recommended when working with strained ring systems (e.g., cyclopropanes, aziridines). (default: True)
        use_macrocycle_torsions: bool
            If True, enables special torsional treatment of macrocycles during embedding.
            Recommended for cyclic peptides or large ring systems (≥12 atoms). (default: True)
        maxiter: int
            The maximum number of iterations allowed for the force field optimization step. (default: 500)
        random_seed: int
            If set to a positive integer, this value is used to seed the random number generator for
            the 3D embedding algorithm. This will ensure reproducible conformations for the same input SMILES.
            This keyword is mainly used for testing. (default: -1)
        """
        geometry = MolecularGeometry.from_smiles(
            smiles,
            force_uff=force_uff,
            default_mmffvariant = default_mmffvariant,
            use_small_ring_torsions = use_small_ring_torsions,
            use_macrocycle_torsions = use_macrocycle_torsions,
            maxiter = maxiter,
            random_seed = random_seed,
        )

        obj = System(name, geometry, charge=charge, spin=spin, box_side=box_side)
        return obj
               

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
        data["Geometry"] = self.geometry.to_dict()
        data["Properties"] = self.properties.to_dict()
        data["Flags"] = self.flags

        with open(path, "w") as jsonfile:
            json.dump(data, jsonfile)

    @property
    def geometry(self) -> MolecularGeometry:
        """
        The `MolecularGeometry` object encoding the geometry of the system       
        """
        return self.__geometry

    @geometry.setter
    def geometry(self, new_geometry: MolecularGeometry) -> None:

        if type(new_geometry) != MolecularGeometry:
            raise TypeError("The geometry attribute must be of type MolecularGeometry")

        elif new_geometry.atomcount == 0:
            raise ValueError("The geometry object cannot be empty or not initialized")

        self.__geometry = new_geometry
        self.__geometry._MolecularGeometry__add_system_reset(self.__on_geometry_change)
        logger.info(f"Geometry changed: clearing properties for {self.name}")
        self.properties = Properties()
    
    @property
    def properties(self) -> Properties:
        """
        The `Properties` class object storing the computed system properties.       
        """
        return self.__properties
    
    @properties.setter
    def properties(self, new_properties: Properties) -> None:
        
        if type(new_properties) != Properties:
            raise TypeError("The properties attribute must be of type `Properties`")

        self.__properties = new_properties
        self.__properties._Properties__add_check_geometry_level_of_theory(self.__check_geometry_level_of_theory)

    @property
    def charge(self) -> int:
        """
        The total charge of the molecular system       
        """
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
        """
        The spin multiplicity of the molecular system       
        """
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
        """
        The length of the side of the simulation box (in Å) used to define periodic boundary conditions.
        If set to `None`, the system is treated as non-periodic.
        """
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

        info += "Center of mass:\n"
        info += "----------------------------------------------\n"
        info += "                x (Å)      y (Å)      z (Å)   \n"
        info += "----------------------------------------------\n"
        info += f" {'':<6}{'':^6}"
        for c in self.geometry.center_of_mass:
            info += "{0:^11}".format(f"{c:.5f}")
        info += "\n"
        info += "----------------------------------------------\n\n"

        info += "Inertia tensor (amu·Å²):\n"
        info += "----------------------------------------------\n"
        info += "                  x          y          z     \n"
        info += "----------------------------------------------\n"
        for column, row in zip(["x", "y", "z"], self.geometry.inertia_tensor):
            info += f" {column:<10}"
            for val in row:
                info += f"{val:>11.5f}"
            info += "\n"
        info += "----------------------------------------------\n\n"
        info += "Principal axes of rotation:\n"
        info += "----------------------------------------------\n"
        info += "                  x          y          z     \n"
        info += "----------------------------------------------\n"
        for column, row in zip(["A", "B", "C"], self.geometry.inertia_eigvecs):
            info += f" {column:<10}"
            for val in row:
                info += f"{val:>11.5f}"
            info += "\n"
        info += "----------------------------------------------\n\n"
        info += f"Rotor type: {self.geometry.rotor_type}\n"
        eigvals = self.geometry.inertia_eigvals
        info += f"Principal moments (amu·Å²):\t{eigvals[0]:.5f}  {eigvals[1]:.5f}  {eigvals[2]:.5f}\n\n"
        rot_const = self.geometry.rotational_constants[0]
        info += f"Rotational constants (cm⁻¹):\t{rot_const[0]:.5f}  {rot_const[1]:.5f}  {rot_const[2]:.5f}\n"
        rot_const = self.geometry.rotational_constants[1]
        info += f"Rotational constants (MHz):\t{rot_const[0]:.5f}  {rot_const[1]:.5f}  {rot_const[2]:.5f}\n\n"
        info += "----------------------------------------------\n\n"

        info += "********************** PROPERTIES *************************\n\n"
        info += f"Geometry level of theory: {self.geometry.level_of_theory_geometry}\n"
        info += (
            f"Electronic level of theory: {self.properties.level_of_theory_electronic}\n"
        )
        info += f"Vibronic level of theory: {self.properties.level_of_theory_vibrational}\n\n"
        info += f"Electronic energy: {self.properties.electronic_energy} Eh\n"
        info += f"Gibbs free energy correction G-E(el): {self.properties.free_energy_correction} Eh\n"
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

    def __iter__(self) -> Generator[System]:
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

        NOTE: the vibrational contributions are included in the electronic component, which
        actually contains the TOTAL energy of the system. Maybe in the future I'll think of
        how to separate the two contributions - LB
        """
        energies = []

        for system in self.systems:
            if system.properties.free_energy_correction is None:
                energies.append(system.properties.electronic_energy)
            else:
                energies.append(
                    system.properties.electronic_energy + system.properties.free_energy_correction
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
