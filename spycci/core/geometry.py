from __future__ import annotations

import numpy as np
import os, sh, shutil

from os.path import isfile
from copy import deepcopy
from typing import Tuple, List, Union, Generator, Optional, Iterator, Callable, TYPE_CHECKING, Dict
from tempfile import mkdtemp
from morfeus import BuriedVolume

from rdkit.Chem import Mol, Atom, Bond, MolFromSmiles, AddHs, MolFromXYZFile, GetAdjacencyMatrix
from rdkit.Chem.rdDistGeom import EmbedMolecule, ETKDGv3
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from rdkit.Chem.rdForceFieldHelpers import (
    UFFOptimizeMolecule,
    MMFFOptimizeMolecule,
    UFFHasAllMoleculeParams,
    MMFFHasAllMoleculeParams,
)

from spycci.constants import atoms_dict, atomic_masses, h, c, amu_to_kg

if TYPE_CHECKING:
    from spycci.systems import System

class MolecularGeometry:
    """
    The `MolecularGeometry` class implements all the functions required to operate on the
    geometric properties of a given molecule or molecular aggregate. The `MolecularGeometry` class
    implements various classmethods capable of constructing an instance of the class from `.xyz`
    files (`from_xyz()`), from SMILES strings (`from_smiles()`) or from a dictionary (`from_dict()`).
    The coordinates of a molecule or its composition can be altered throug `append` operations or
    through the setter `set_atoms()` and `set_coordinates()` methods. Coordinates and atom list can
    be accessed, in the form of a read-only deepcopy using the `get_atoms()` and `get_coordinates()`
    methods.

    Attributes
    ----------
    level_of_theory_geometry: str
        The level of theory at which the geometry has been obtained.
    """

    def __init__(self) -> None:
        self.__atomcount: int = 0
        self.__atoms: List[str] = []
        self.__coordinates: List[np.ndarray] = []

        self.__inertia_tensor: Optional[np.ndarray] = None
        self.__inertia_eigvals: Optional[np.ndarray] = None
        self.__inertia_eigvecs: Optional[np.ndarray] = None
        self.__rotor_type: Optional[str] = None
        self.__rotational_constants: Optional[tuple] = None
        
        self.__adjacency_matrix: Optional[np.ndarray] = None
        self.__bond_type_matrix: Optional[np.ndarray] = None

        self.level_of_theory_geometry: Optional[str] = None

        # Define a listener to reset System on geometry change
        self.__system_reset: System.__on_geometry_change = None
    

    def __len__(self) -> int:
        return self.__atomcount
    
    def __deepcopy__(self, memo) -> MolecularGeometry:
        "Overload of the deepcopy funtion to safely remove listener reference"
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj

        for attr_name, attr_value in self.__dict__.items():
            setattr(obj, attr_name, deepcopy(attr_value, memo))
        
        obj.__system_reset = None

        return obj

    def append(self, atom: str, coordinates: Union[List[float], np.ndarray]) -> None:
        """
        The append function allows the user to add the position of a new atom belonging to
        the molecule.

        Arguments
        ---------
        atom: str
            The symbol of the atom
        coordinates: Union[List[float], np.ndarray]
            The list or numpy array of 3 floating point values indicating the cartesian
            position of the atom in the tridimensional space

        Raises
        ------
        ValueError
            Exception raised if the atom does not represent a valid element or if the coordinates
            vector does not match the requirements
        """

        if atom not in atoms_dict.values():
            raise ValueError(f"The symbol {atom} is not a valid element")

        if len(coordinates) != 3:
            raise RuntimeError(
                f"The coordinate vector must contain 3 floating poin coordinates"
            )

        self.__clear_properties()
        self.__call_system_reset()
        self.__atomcount += 1
        self.__atoms.append(atom)
        self.__coordinates.append(np.array(coordinates))
    
    @property
    def atoms(self) -> List[str]:
        """
        The list of atoms/elements in the molecule. Please beware that the obtained
        atoms list is provided as a deepcopy and not as a reference. If you want to
        set new atoms names, please use the `set_atoms()` function.

        Returns
        -------
        List[str]
            The list of strings representing, in order, the symbols of the atoms in the molecule
        """
        return deepcopy(self.__atoms)
    
    def set_atoms(self, atoms: List[str]) -> None:
        """
        Set a new list of atoms composing the molecule. Please beware that the new
        atom list length must match the number of atoms/coordinates already stored.

        Arguments
        ---------
        List[str]
            The list element symbols encoding the atoms in the molecule.
        
        Raises
        ------
        ValueError
            Exception raised if the number of atoms does not match the one currently stored.
        """
        # Check that the length of the newly provided atom list is coherent with the stored coordinates.
        if len(atoms) != self.atomcount:
            raise ValueError(f"The new length of the atom list ({len(atoms)}) cannot be different from the current atomcount ({self.atomcount}).")

        # Clear all stored properties and call the `System` class listener and update atoms list
        self.__clear_properties()
        self.__call_system_reset()
        self.__atoms = atoms

    @property
    def coordinates(self) -> List[np.ndarray]:
        """
        The list of coordinates of each atom in the molecule. Please beware that the obtained
        coordinates list is provided as a deepcopy and not as a reference. If you want to
        set new coordinates values please use the `set_coordinates()` function.

        Returns
        -------
        List[np.ndarray]
            The list of numpy arrays representing, in order, the 3D position of each atom
            in the molecule
        """
        return deepcopy(self.__coordinates)
    
    def set_coordinates(self, coordinates: List[np.ndarray]) -> None:
        """
        Set a new list of coordinates for the atoms in the molecule. Please beware that the new
        coordinates must match the number of atoms already stored.

        Arguments
        ---------
        List[np.ndarray]
            The list of numpy arrays encoding the 3D position of each atom in the molecule.
        
        Raises
        ------
        ValueError
            Exception raised if the number of coordinates does not match the number of atoms in the
            molecule or if the length of each position vector is different from 3.
        """
        # Check that the length of the newly provided coordinate list is coherent with the stored atoms.
        if len(coordinates) != self.atomcount:
            raise ValueError(f"The new length of the coorinates list ({len(coordinates)}) cannot be different from the current atomcount ({self.atomcount}).")
        
        # Convert the input as a List[np.ndarray] to account for the user passing a list of lists
        coords = [np.array(v) for v in coordinates]
        for v in coords:
            if len(v) != 3:
                raise ValueError("Atomic coordinate vectors must be of length 3.")

        # Clear all stored properties and call the `System` class listener
        self.__clear_properties()
        self.__call_system_reset()
        self.__coordinates = coords

    @classmethod
    def from_xyz(cls, path: str) -> MolecularGeometry:
        """
        The functions returns a fully initialized `MolecularGeometry` class containing the
        coordinates indicated in the `.xyz` file located in the indicated path.

        Arguments
        ---------
        path: str
            A string indicating the path to a valid `.xyz` file

        Returns
        -------
        MolecularGeometry
            The `MolecularGeometry` object containing the coordinates encoded in the `.xyz` file
        """
        obj = cls()
        obj.load_xyz(path)
        return obj

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        force_uff: bool = False,
        default_mmffvariant: str = "MMFF94",
        use_small_ring_torsions: bool = True,
        use_macrocycle_torsions: bool = True,
        maxiter: int = 500,
        random_seed: int = -1,
    ) -> MolecularGeometry:
        """
        The function returns a fully initialized `MolecularGeometry` object from the given SMILES string.
        The returned geometry is generated by embedding the molecule in 3D using the ETKDGv3 algorithm,
        followed by energy minimization using MMFF or UFF force fields, depending on availability
        and user preference. The operations are carried out using the tools defined in the RDKit package.

        Arguments
        ---------
        smiles: str
            A valid SMILES string representing the molecular structure to be converted into 3D coordinates.
        force_uff: bool
            If True, the geometry optimization will use the UFF force field even when MMFF parameters are available.
            This can be useful when consistency across a dataset using UFF is desired. (default: False)
        default_mmffvariant: str,
            Specifies the MMFF variant to use when MMFF is selected. Valid options are "MMFF94" or "MMFF94s".
            "MMFF94s" is generally recommended for more accurate static geometries.
        use_small_ring_torsions: bool
            If True, enables special torsional sampling for small rings during 3D embedding.
            Recommended when working with strained ring systems (e.g., cyclopropanes, aziridines). (default: True)
        use_macrocycle_torsions: bool
            If True, enables special torsional treatment of macrocycles during embedding.
            Recommended for cyclic peptides or large ring systems (≥12 atoms). (default: True)
        maxiter: int
            The maximum number of iterations allowed for the force field optimization step. (default=500)
        random_seed: int
            If set to a positive integer, this value is used to seed the random number generator for
            the 3D embedding algorithm. This will ensure reproducible conformations for the same input SMILES.
            This keyword is mainly used for testing. (default: -1)

        Returns
        -------
        MolecularGeometry
            A `MolecularGeometry` object containing atomic symbols and optimized 3D coordinates derived from the
            input SMILES string.
        """
        mol: Mol = MolFromSmiles(smiles)
        mol = AddHs(mol)

        params = ETKDGv3()
        params.randomSeed = random_seed
        params.useSmallRingTorsions = use_small_ring_torsions
        params.useMacrocycleTorsions = use_macrocycle_torsions

        EmbedMolecule(mol, params)

        if MMFFHasAllMoleculeParams(mol) and force_uff is False:
            MMFFOptimizeMolecule(mol, mmffVariant=default_mmffvariant, maxIters=maxiter)
        else:
            if UFFHasAllMoleculeParams(mol) is False:
                raise RuntimeError("RDKit UFFOptimize has not all the required molecular parameters")
            UFFOptimizeMolecule(mol, maxIters=maxiter)

        atoms: Iterator[Atom] = mol.GetAtoms()
        conf = mol.GetConformer()

        obj = cls()
        obj.__atomcount = len(atoms)

        for atom in atoms:
            pos = conf.GetAtomPosition(atom.GetIdx())
            obj.__atoms.append(atom.GetSymbol())
            obj.__coordinates.append(np.array([pos.x, pos.y, pos.z]))

        return obj

    @classmethod
    def from_dict(cls, data: dict) -> MolecularGeometry:
        """
        Construct a MolecularGeometry object from the data encoded in a dictionary.

        Arguments
        ---------
        data: dict
            The dictionary containing the class attributes

        Returns
        -------
        MolecularGeometry
            The fully initialized MolecularGeometry object
        """
        obj = cls()
        obj.__atomcount = data["Number of atoms"]
        obj.__atoms = data["Elements list"]
        obj.__coordinates = [np.array(v) for v in data["Coordinates"]]
        obj.level_of_theory_geometry = data["Level of theory geometry"]
        return obj

    def to_dict(self) -> dict:
        """
        Generates a dictionary representation of the class. The obtained dictionary can be
        saved and used to re-load the object using the built-in `from_dict` class method.

        Returns
        -------
        dict
            The dictionary listing, with human friendly names, the attributes of the class
        """
        data = {}
        data["Number of atoms"] = self.__atomcount
        data["Elements list"] = self.__atoms
        data["Coordinates"] = [list(v) for v in self.__coordinates]
        data["Level of theory geometry"] = self.level_of_theory_geometry
        return data

    def load_xyz(self, path: str) -> None:
        """
        Imports the coordinates of the molecule from a path pointing to a valid `.xyz` file

        Arguments
        ---------
        path: str
            The path to the `.xyz` from which the coordinates must be loaded

        Raises
        ------
        ValueError
            Exception raised if the path given does not point to a valid file
        RuntimeError
            Exception raised if an error occurs while loading the data from the file
        """

        # Clean all the variables
        self.__atomcount = 0
        self.__atoms = []
        self.__coordinates = []

        # Clear all stored properties and call the `System` class listener
        self.__clear_properties()
        self.__call_system_reset()

        # Check if the given path points to a valid file
        if not isfile(path):
            raise ValueError(f"The path {path} does not point to a valid file.")

        # Open the file in read mode
        with open(path, "r") as file:

            # Read the whole file and count both the number of lines and terminal empty lines
            nlines, offset = 0, 0
            for line in file:
                nlines += 1

                if line == "\n":
                    offset += 1
                else:
                    offset = 0

            file.seek(0)

            # Extract the number of atoms from the first line and compute the beginning of
            # the last xyz coordinate block (required when operating on trajectories)
            self.__atomcount = int(file.readline())
            beginning = nlines - (self.__atomcount + offset + 2)

            file.seek(0)

            # Read the file line by line starting from the beginning
            line: str = ""
            for n, line in enumerate(file):

                # Split the line to seprate the various fields
                sline = line.split()

                # Discart all the lines before the last block and the comment line
                if n < beginning or n == beginning + 1:
                    continue

                # Check that the block begins with the expected atomcount to confirm
                elif n == beginning:
                    if len(sline) != 1 or int(sline[0]) != self.__atomcount:
                        raise RuntimeError(
                            "The beginning of the xyz block does not match the expected atomcount"
                        )
                    else:
                        continue

                # Discard all the trailing empty lines
                elif line == "\n":
                    continue

                # If the file contains atomic numbers instead of symbols convert them into
                # the latter, else directly read the symbol
                try:
                    atom = atoms_dict[int(sline[0])]
                except:
                    atom = sline[0]

                # Check that the element exists between the list of known elements
                if atom not in atoms_dict.values():
                    raise RuntimeError(f"The symbol {atom} is not a valid element")

                # Append the atom and its coordinates to the class member variables
                self.__atoms.append(atom)
                self.__coordinates.append(np.array([float(x) for x in sline[1:4]]))

            # Check that the lengths of the array match the number of atoms expected
            if (
                len(self.__atoms) != self.__atomcount
                or len(self.__coordinates) != self.__atomcount
            ):
                raise RuntimeError("Mismatch between the atom count and the loaded data")

    def write_xyz(self, path: str, comment: str = "") -> None:
        """
        Exports the coordinates to a `.xyz` file located at a given path

        Arguments
        ---------
        path: str
            A valid path in which the `.xyz` file must be saved
        comment: str
            An optional comment that can be saved in the `.xyz` file
        """
        with open(path, "w") as file:

            file.write(f"{self.__atomcount}\n")
            file.write(f"{comment}\n")

            for atom, position in zip(self.__atoms, self.__coordinates):
                file.write(atom)
                for xi in position:
                    file.write(f"    {xi:.10f}")
                file.write("\n")

    @property
    def atomcount(self) -> int:
        """
        The number of atoms in the molecule

        Returns
        -------
        int
            The number of atoms in the molecule
        """
        return self.__atomcount

    @property
    def atomic_numbers(self) -> List[int]:
        """
        The ordered list of the atomic numbers of each atom in the molecule

        Returns
        -------
        List[int]
            The list of integers atomic numbers associated with each atom in the molecule
        """
        ATOMIC_NUMBERS = {v: k for k, v in atoms_dict.items()}
        return [ATOMIC_NUMBERS[element] for element in self.__atoms]

    @property
    def mass(self) -> float:
        """
        The mass of the molecule in atomic mass units computed using the average atomic weights.

        Returns
        -------
        float
            The molecular mass in atomic mass units.
        """
        mass = 0
        for atom in self.__atoms:
            mass += atomic_masses[atom]
        return mass

    @property
    def center_of_mass(self) -> np.ndarray:
        """
        The center of mass of the molecule in Angstrom.

        Returns
        -------
        np.ndarray
            The numpy array containing the 3 cartesian coordinates of the center of mass
            of the molecule in Angstrom.
        """
        com = np.zeros(3)
        for atom, position in zip(self.__atoms, self.__coordinates):
            mass = atomic_masses[atom]
            com += mass * position
        com /= self.mass
        return com

    @property
    def inertia_tensor(self) -> np.ndarray:
        """
        The inertia tensor of the molecule (in amu·Å²) calculated relative to the
        molecular center of mass, using atomic masses (in atomic mass units) and
        cartesian coordinates (in Ångström).

        Returns
        -------
        np.ndarray
            The inertia tensor of the molecule in amu·Å² as a numpy array of shape (3, 3).
        """
        if self.__inertia_tensor is None:
            self.__calculate_inertia()
        return deepcopy(self.__inertia_tensor)

    @property
    def inertia_eigvals(self) -> np.ndarray:
        """
        The principal moments of inertia (IA, IB, IC) (in amu·Å²) computed as eigenvalues
        of the inertia tensor.

        Returns
        -------
        np.ndarray
            The principal moments of inertia (IA, IB, IC) in amu·Å² as a numpy array of shape (3).
        """
        if self.__inertia_eigvals is None:
            self.__calculate_inertia()
        return deepcopy(self.__inertia_eigvals)

    @property
    def inertia_eigvecs(self) -> np.ndarray:
        """
        The principal axes of rotation computed as eigenvectors of the inertia tensor.

        Returns
        -------
        np.ndarray
            The principal axes of rotation as a numpy array of shape (3, 3).
        """
        if self.__inertia_eigvecs is None:
            self.__calculate_inertia()
        return deepcopy(self.__inertia_eigvecs)

    @property
    def rotor_type(self) -> str:
        """
        The type of molecular rigid rotor determined based on the relative magnitudes of the
        principal moments of inertia:

            * Linear rotor:           IA ≈ 0 and IB ≈ IC
            * Spherical top:          IA ≈ IB ≈ IC
            * Oblate symmetric top:   IA ≈ IB < IC (disc-shaped)
            * Prolate symmetric top:  IA < IB ≈ IC (cigar-shaped)
            * Asymmetric top:         all moments different

        Returns
        -------
        str
            Type of molecular rigid rotor.
        """
        if self.__rotor_type is None:
            self.__calculate_inertia()
        return self.__rotor_type

    @property
    def rotational_constants(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        The rotational constants (A, B, C)  of the molecule in cm⁻¹ and MHz defined as:

        .. math::
            B_\alpha := \frac{\hbar^2}{2 I_\alpha}

        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple of two numpy arrays of shape (3) containing the rotational constants (A, B, C)
            of the molecule expressed in cm⁻¹ (first) and in MHz (second).
        """
        if self.__rotational_constants is None:
            self.__calculate_inertia()
        return deepcopy(self.__rotational_constants)
    
    @property
    def adjacency_matrix(self) -> np.ndarray:
        """
        The adjacency matrix encoding the molecular connectivity. The matrix is a N x N square
        and symmetrical matrix (with N the total number of atoms in the geometry) encoding wherher
        two atoms are bonded or not. If the atom i and j are connected, the [i,j] and [j, i] matrix
        elements will be set to one, otherwise zero. The matrix is internally generated using RDKit.

        Returns
        -------
        np.ndarray
            The NxN (with N the total number of atoms) symmetrical matrix encoding the connectivity
            of the molecule
        """
        if self.__adjacency_matrix is None:
            self.__generate_connectivity()
        
        return deepcopy(self.__adjacency_matrix)
    
    @property
    def bond_type_matrix(self) -> np.ndarray:
        """
        The bond type matrix encoding the molecular connectivity and the order/type of the bonds
        connecting each pair of atoms. The matrix is a N x N square and symmetrical matrix (with N
        the total number of atoms in the geometry) encoding whether two atoms are bonded or not and
        the order/type of the bond between them. If the atom i and j are connected, the [i,j] and 
        [j, i] matrix elements will be set to the float value representing the bond order/type (i.e.
        1.0 for single, 2.0 for double, 3.0 for triple, 1.5 for aromatic bonds), otherwise zero if they
        are not connected. The matrix is internally generated using the adjacency matrix and bond types
        generated by RDKit. Note that the bond orders represented here are purely topological and 
        do NOT correspond to quantum-chemically derived bond indices (e.g., Wiberg or Mayer bond orders).

        Returns
        -------
        np.ndarray
            The NxN (with N the total number of atoms) symmetrical matrix encoding the connectivity
            of the molecule
        """
        if self.__bond_type_matrix is None:
            self.__generate_connectivity()
        
        return deepcopy(self.__bond_type_matrix)


    def buried_volume_fraction(
        self,
        site: int,
        radius: float = 3.5,
        density: float = 0.001,
        include_hydrogens: bool = True,
        excluded_atoms: List[int] = None,
        radii: List[float] = None,
        radii_type: str = "bondi",
        radii_scale: float = 1.17,
    ):
        """
        Computes the buried volume fraction around a given site. The functions adopts the implementation
        provided in the morfeus python package (https://kjelljorner.github.io/morfeus/buried_volume.html).

        Arguments
        ---------
        site: int
            The index (starting the numeration from zero) of the reactive atom around which
            the buried volume must be computed
        radius: float
            The radius (in Angstrom) of the sphere in which the buried volume should be computed
        density: float
            The volume (in Angstrom^3) per point in the sphere used in the calculation
        include_hydrogens: bool
            If set to True (default) will consider the hydrogen atoms in the calculation
        excluded_atoms: List[int]
            The list of indices (starting the numeration from zero) of the atom that should be
            excluded from the compuitation of the buried volume (default: None)
        radii: List[float]
            The custom list of atomic radii to be used in the computation (default: None).
            If set to a value different from `None`, will override the `radii_type` and
            `radii_scale` options.
        radii_type: str
            Type of radii to use in the calculation. The available options are `alvarez`,
            `bondi`, `crc` or `truhlar`.
        radii_scale: float
            Scaling factor for the selected radii_type

        Raises
        ------
        ValueError
            Exception raised if either the index of the selected atoms are out of bounds or
            if the radii type does not match any of the presets.

        Returns
        -------
        float
            The fraction of buried volume of the sphere (between 0 and 1)
        """

        if site < 0 or site >= self.atomcount:
            raise ValueError(
                f"The site index {site} is out of bounds (current atomcount: {self.atomcount})"
            )

        if excluded_atoms != None:
            for idx in excluded_atoms:
                if idx < 0 or idx >= self.atomcount:
                    raise ValueError(
                        f"The excluded atom index {idx} is out of bounds (current atomcount: {self.atomcount})"
                    )

        if radii_type not in ["alvarez", "bondi", "crc", "truhlar"]:
            raise ValueError(f"The radii definition {radii_type} is not available at this time.")

        if radii is not None:
            if len(radii) != self.atomcount:
                raise ValueError("The length of the radii list must match the number of atoms in the molecule")

        bv = BuriedVolume(
            deepcopy(self.__atoms),
            deepcopy(self.__coordinates),
            site + 1,
            excluded_atoms=(
                [idx + 1 for idx in excluded_atoms]
                if excluded_atoms is not None
                else None
            ),
            radii=radii,
            radii_type=radii_type,
            radii_scale=radii_scale,
            density=density,
            include_hs=include_hydrogens,
            radius=radius,
        )

        return bv.fraction_buried_volume
    
    ####################################################################################
    #                                 HELPER FUNCTIONS                                 #
    ####################################################################################

    def __add_system_reset(self, listener: System.__on_geometry_change) -> None:
        """
        Add a reference to a System reset function

        Argument
        --------
        listener: System.__on_geometry_change
            The method of the `System` object handling a change in geometry
        """
        self.__system_reset = listener
    
    def __call_system_reset(self) -> None:
        "If set, call the system (owner) reset listener"
        if self.__system_reset is not None:
            self.__system_reset()

    def __clear_properties(self) -> None:
        """
        Clears the level of theory and all the structure related properties
        that may have been cached for a previously defined molecular structure.
        """
        self.__inertia_tensor = None
        self.__inertia_eigvals = None
        self.__inertia_eigvecs = None
        self.__rotor_type = None
        self.__rotational_constants = None
        self.__adjacency_matrix = None
        self.__bond_type_matrix = None

        self.level_of_theory_geometry = None
    
    def __generate_mol(self, charge: int = 0) -> Mol:
        """
        Generates `rdkit.Chem.Mol` object from the stored molecular geometry. Connectivity
        is automatically generated using the `rdkit.Chem.rdDetermineBonds.DetermineBonds`
        function.

        Arguments
        ---------
        charge: int
            The charge of the molecule encoded by the current geometry (default: 0, neutral)

        Returns
        -------
        rdkit.Chem.Mol
            The `Mol` object of the RDKit library representing the molecular object.
        """
        mol = None
        tdir = mkdtemp(prefix="RDKitWorkdir_", suffix=f"_MolGen", dir=os.getcwd() )

        with sh.pushd(tdir):

            self.write_xyz(f"molecule.xyz")
            raw_mol = MolFromXYZFile("molecule.xyz")
            mol = Mol(raw_mol)

            shutil.rmtree(tdir)

        DetermineBonds(mol,charge=charge, embedChiral=True)

        return mol
    
    def __generate_connectivity(self) -> None:
        """
        Given the current molecular geometry, generate connectivity data using RDKit. The function
        internally sets the `self.__adjacency_matrix` and the `self.__bond_type_matrix` variables.
        """
        
        mol = self.__generate_mol()
        self.__adjacency_matrix = GetAdjacencyMatrix(mol)

        btype_matrix = np.zeros((self.atomcount, self.atomcount), dtype=float)

        bond: Bond = None
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            order = bond.GetBondTypeAsDouble()
            btype_matrix[i, j] = order
            btype_matrix[j, i] = order
        
        self.__bond_type_matrix = btype_matrix
        

    def __calculate_inertia(self) -> None:
        """
        Calculate and set the inertia tensor, its eigenvalues and eigenvectors, rotor type,
        and rotational constants of the molecule.

        The inertia tensor is calculated relative to the molecular center of mass,
        using atomic masses (in atomic mass units) and cartesian coordinates
        (in Ångström). The eigenvalues of the tensor correspond to the principal
        moments of inertia (IA, IB, IC).

        The rotor type is determined based on the relative magnitudes of the
        principal moments:

            * Linear rotor:           IA ≈ 0 and IB ≈ IC
            * Spherical top:          IA ≈ IB ≈ IC
            * Oblate symmetric top:   IA ≈ IB < IC (disc-shaped)
            * Prolate symmetric top:  IA < IB ≈ IC (cigar-shaped)
            * Asymmetric top:         all moments different
        
        The rotational constants are provided in both cm⁻¹ and MHz.
        """
        xyz_centered = np.subtract(self.__coordinates, self.center_of_mass)
        masses = np.array([atomic_masses[atom] for atom in self.__atoms])

        x, y, z = xyz_centered.T

        Ixx = np.sum(masses * (y**2 + z**2))
        Iyy = np.sum(masses * (x**2 + z**2))
        Izz = np.sum(masses * (x**2 + y**2))
        Ixy = -np.sum(masses * x * y)
        Iyz = -np.sum(masses * y * z)
        Ixz = -np.sum(masses * x * z)

        self.__inertia_tensor = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz]
        ])

        self.__inertia_eigvals, self.__inertia_eigvecs = np.linalg.eigh(self.__inertia_tensor)

        eigvals_kgm2 = self.__inertia_eigvals * amu_to_kg / 1.0e20
        
        rot_const_cm, rot_const_mhz = [], []
        for eigval in eigvals_kgm2:

            if eigval == 0.:
                rot_const_cm.append(None)
                rot_const_mhz.append(None)

            else:
                value = h / (8 * np.pi**2 * c * 100 * eigval)
                rot_const_cm.append(value)
                rot_const_mhz.append(value * c / 1.0e4)           
                
        self.__rotational_constants = (np.array(rot_const_cm), np.array(rot_const_mhz))

        tol=1e-3
        IA, IB, IC = self.__inertia_eigvals
        if IA < tol and abs(IB - IC) < tol:
            self.__rotor_type = "linear rotor"
        elif abs(IA - IB) < tol and abs(IB - IC) < tol:
            self.__rotor_type = "spherical top"
        elif abs(IA - IB) < tol and abs(IC - IB) > tol:
            self.__rotor_type = "oblate symmetric top"
        elif abs(IB - IC) < tol and abs(IA - IB) > tol:
            self.__rotor_type = "prolate symmetric top"
        else:
            self.__rotor_type = "asymmetric top"