from __future__ import annotations

from os.path import isfile
from typing import List, Tuple
from copy import deepcopy

import numpy as np

from spycci.constants import atoms_dict, bohr_to_angstrom


class Cube:
    """
    Simple Cube class allowing the loading, saving and manipulation of cube files.
    """

    def __init__(self) -> None:
        self.__atomcount: int = None  # Number of atoms in the molecule
        self.__origin: np.ndarray = None  # Position of the volumetric data
        self.__nvoxels: List[int] = []  # Number of voxels for each dimension
        self.__axes: List[np.ndarray] = []  # List of the three axes vectors
        self.__atomic_numbers: List[int] = []  # List of atomic numbers of the atoms
        self.__atomic_charges: List[float] = []  # List of atomic "charges" of the atoms
        self.__coordinates: List[np.ndarray] = []  # List of coordinates of each atom
        self.__cube: np.ndarray = None  # The volumetric data in cube format
        self.__is_bohr: bool = None     # If True the cube is written in atomic units else angstroms

    @classmethod
    def from_file(cls, path: str) -> Cube:
        """
        Class method capable of constructing a Cube object from a Gaussian formatted cube file

        Parameters
        ----------
        path: str
            A string containing a valid path to a Gaussian cube file.

        Raises
        ------
        ValueError
            Exception raised when the path does not point to a valid file.

        Returns
        -------
        Cube
            An instance of the `Cube` class containing all the data loaded form the indicated
            path.
        """
        if not isfile(path):
            raise ValueError(f"The path '{path}' does not point to a valid file")

        obj = Cube()
        with open(path, "r") as file:

            # Discard the first two comment lines
            _ = file.readline()
            _ = file.readline()

            # Read the line containing the number of atoms and the position of the origin
            data = file.readline().split()
            obj.__atomcount = int(data[0])
            obj.__origin = np.array([float(value) for value in data[1::]])
            
            # ************************* TEMPORARY PATCH *************************
            
            # Check if atomcount is negative as in a orca MO cube file
            is_orca_mo = False
            if obj.__atomcount < 0:
                is_orca_mo = True
                obj.__atomcount *= -1

            # *******************************************************************

            # Read the number of voxel along each axis and the axis vector
            for _ in range(3):
                data = file.readline().split()
                obj.__nvoxels.append(int(data[0]))
                obj.__axes.append(np.array([float(value) for value in data[1::]]))
            
            # Check if values are in atomic units (bohr) or Angstrom (also check consistency)
            if all([n<0 for n in obj.__nvoxels]):
                obj.__is_bohr = False
                obj.__nvoxels = [-n for n in obj.__nvoxels]

            elif all([n>0 for n in obj.__nvoxels]):
                obj.__is_bohr = True
            
            else:
                raise RuntimeError("Unexpected behavior in the number of voxels.")

            # Read the section containing the atomic positions
            for _ in range(obj.__atomcount):
                data = file.readline().split()
                obj.__atomic_numbers.append(int(data[0]))
                obj.__atomic_charges.append(float(data[1]))
                obj.__coordinates.append(
                    np.array([float(value) for value in data[2::]])
                )
            
            # ************************* TEMPORARY PATCH *************************
            
            if is_orca_mo is True:

                file_position = file.tell()
                
                data = file.readline().split()
                if len(data) != 2:
                    file.seek(file_position)

            # *******************************************************************

            # Read the rest of the file and reshape it according to the number of voxels
            data_str = file.read().split()
            buffer = np.fromiter(map(float, data_str), dtype=np.float32)
            obj.__cube = buffer.reshape(tuple(obj.__nvoxels))

        return obj

    def save(self, path: str, comment_1st: str = "", comment_2nd: str = "") -> None:
        """
        Class method capable of saving a Cube object in a Gaussian formatted cube file

        Parameters
        ----------
        path: str
            A string containing a valid path to the destination file.
        comment_1st: str
            The string encoding the first comment line
        comment_2st: str
            The string encoding the second comment line
        """
        with open(path, "w") as file:

            # Write the first two comment lines
            file.write(f"{comment_1st}\n")
            file.write(f"{comment_2nd}\n")

            # Write the line containing the number of atoms and the position of the origin
            line = f"{self.__atomcount}    "
            line += "    ".join([f"{v:.6e}" for v in self.__origin])
            file.write(f"{line}\n")

            # Write the number of voxel along each axis and the axis vector
            for nv, axis in zip(self.__nvoxels, self.__axes):
                line = f"{nv}    " if self.__is_bohr else f"{-nv}    "
                line += "    ".join([f"{v:.6e}" for v in axis])
                file.write(f"{line}\n")

            # Write the section containing the atomic positions
            for atomic_number, charge, coords in zip(
                self.__atomic_numbers,
                self.__atomic_charges,
                self.__coordinates,
            ):
                line = f"{atomic_number}    {charge:.6e}    "
                line += "    ".join([f"{c:.6e}" for c in coords])
                file.write(f"{line}\n")

            lines = []
            for i in range(self.__nvoxels[0]):
                for j in range(self.__nvoxels[1]):

                    chunks = []
                    for k in range(self.__nvoxels[2]):

                        chunks.append(f"{self.__cube[i][j][k]:.6e}")
                        if k % 6 == 5:
                            lines.append("    ".join(chunks))
                            chunks = []

                    if len(chunks) != 0:
                        lines.append("    ".join(chunks))

            file.write("\n".join(lines) + "\n")

    def __validate(self, other: Cube, rtol: float = 1e-2) -> None:
        """
        Simple validation function to compare the Cube objects before performing an operation.
        Given an input cube the functions compares the atom list and grid parameters to ensure
        that the cubes are referred to the same molecule and have the same orientation an spacing.

        Paremeters
        ----------
        other: Cube
            The cube provided for the operation
        rtol: float
            The relative tollerance to consider two floating point number equivalents.

        Raises
        ------
        ValueError
            Exception raised if the two cube objects are not compatible and, as such, a binary
            operation cannot be performed.
        """
        if not all(
            [
                self.__atomcount == other.__atomcount,
                self.__nvoxels == other.__nvoxels,
                np.allclose(self.origin, other.origin, rtol=rtol),
                np.allclose(self.axes, other.axes, rtol=rtol),
                np.allclose(self.coordinates, other.coordinates, rtol=rtol),
            ]
        ):
            raise ValueError(
                "Cannot perform the operation, the cubes object are not compatible"
            )

    def __getitem__(self, key: Tuple[int]) -> float:
        """
        Given a 3 index tuple (i, j, k) returns the value of the voxel at the position
        `i` along the first axis, `j` alomg the second axis and `k` along the third one.
        For the usual case of a cartesian cube i, j, k map x, y, z respectively.

        Arguments
        ---------
        key: Tuple[int]
            The tuple of 3 values encoding the position of the required voxel.
            Slicing according to numpy syntax is accepted.

        Raises
        ------
        ValueError
            Exception raised if the `key` argument is invalid.

        Returns
        -------
        float
            The cube value at the selected voxel.
        """
        if len(key) != 3:
            raise ValueError(
                "The key tuple to access cube coodinates bust have length 3."
            )
        return self.__cube[key]

    def __add__(self, other: Cube) -> Cube:
        """
        Overload of the addition (+) operator to sum two compatible Cube objects

        Parameters
        ----------
        other: Cube
            the Cube to sum

        Returns
        -------
        Cube
            the Cube object having the cube data equal to the sum of the two parent cubes
        """
        self.__validate(other)
        obj = deepcopy(self)
        obj.__cube += other.__cube
        return obj

    def __sub__(self, other: Cube) -> Cube:
        """
        Overload of the subtraction (-) operator to subtract two compatible Cube objects

        Parameters
        ----------
        other: Cube
            the Cube to subtract

        Returns
        -------
        Cube
            the Cube object having the cube data equal to the difference of the two parent cubes
        """
        self.__validate(other)
        obj = deepcopy(self)
        obj.__cube -= other.__cube
        return obj

    def __mul__(self, other: Cube) -> Cube:
        """
        Overload of the multiplication (*) operator to multiply two compatible Cube objects

        Parameters
        ----------
        other: Cube
            the Cube to multiply

        Returns
        -------
        Cube
            the Cube object having the cube data equal to the product of the two parent cubes
        """
        obj = deepcopy(self)
        self.__validate(other)
        obj.__cube *= other.__cube
        return obj

    def __div__(self, other: Cube) -> Cube:
        """
        Overload of the division (/) operator to divide two compatible Cube objects

        Parameters
        ----------
        other: Cube
            the Cube to be used as a divider

        Returns
        -------
        Cube
            the Cube object having the cube data equal to the quotient of the two parent cubes
        """
        obj = deepcopy(self)
        self.__validate(other)
        obj.__cube /= other.__cube
        return obj

    def scale(self, factor: float) -> Cube:
        """
        Returns a version of the cube multiplied by a scale factor

        Parameters
        ----------
        factor: float
            The scale factor to be applied to the cube

        Returns
        -------
        Cube

        """
        obj = deepcopy(self)
        obj.__cube *= factor
        return obj

    @property
    def atomcount(self) -> int:
        """
        The number of atoms in the molecule to which the cube is referred.

        Returns
        -------
        int
            The number of atoms in the molecule
        """
        return self.__atomcount

    @property
    def origin(self) -> np.ndarray:
        """
        The position of the origin of the volumetric data in Angstrom.

        Retuns
        ------
        np.ndarray
            The 3 coordinate vector encoding the position of the origin.
        """
        if self.__is_bohr is True:
            return self.__origin * bohr_to_angstrom
        else:
            return self.__origin
    
    @property
    def origin_bohr(self) -> np.ndarray:
        """
        The position of the origin of the volumetric data in atomic units (Bohr).

        Retuns
        ------
        np.ndarray
            The 3 coordinate vector encoding the position of the origin.
        """
        if self.__is_bohr is True:
            return self.__origin
        else:
            return self.__origin / bohr_to_angstrom

    @property
    def nvoxels(self) -> List[int]:
        """
        The number of voxel in the cube file along each dimension. If the sign
        of the number of voxels in a dimension is positive then the units are Bohr,
        if negative then Angstroms.

        Returns
        -------
        List[int]
            The list of 3 integer values encoding the number of voxels used
            to map the space along the three axes directions.
        """
        return self.__nvoxels

    @property
    def axes(self) -> List[np.ndarray]:
        """
        The list of the 3 axes coordinates (as the side of each voxel) in Angstrom.

        Returns
        -------
        List[np.ndarray]
            The list of 3 `np.ndarray` vectors of length 3 encoding the axis coordinates.
        """
        if self.__is_bohr is True:
            return [x * bohr_to_angstrom for x in self.__axes]
        else:
            return self.__axes
    
    @property
    def axes_bohr(self) -> List[np.ndarray]:
        """
        The list of the 3 axes coordinates (as the side of each voxel) in atomic units (Bohr).

        Returns
        -------
        List[np.ndarray]
            The list of 3 `np.ndarray` vectors of length 3 encoding the axis coordinates.
        """
        if self.__is_bohr is True:
            return self.__axes
        else:
            return [x / bohr_to_angstrom for x in self.__axes]    

    @property
    def atomic_numbers(self) -> List[int]:
        """
        The list of the atomic numbers of each atom in the molecule to which the cube is referred to.

        Returns
        -------
        List[int]
            The list of atomic numbers of each atom.
        """
        return self.__atomic_numbers

    @property
    def atoms(self) -> List[str]:
        """
        The list of the atoms in the molecule to which the cube is referred to.

        Returns
        -------
        List[str]
            The list of atoms (element symbols).
        """
        return [atoms_dict[n] for n in self.__atomic_numbers]

    @property
    def charges(self) -> List[float]:
        """
        The charges associated with each atom in the molecule.
        BEWARE: This field is often overwritten with other properties. Check the
        origin of the cube file to make sure what the content of the charge field is.

        Returns
        -------
        List[float]
            The list of charges associated to each atom in the cube
        """
        return self.__atomic_charges

    @charges.setter
    def charges(self, new_charges: List[float]) -> None:
        if len(new_charges) != self.__atomcount:
            raise ValueError(
                f"Cannot set charges, {len(new_charges)} values given for a {self.__atomcount} atom molecule"
            )
        self.__atomic_charges = new_charges

    @property
    def coordinates(self) -> List[np.ndarray]:
        """
        The list of coordinates in Angstrom encoding the position of each atom in the molecule.

        Returns
        -------
        List[np.ndarray]
            The list of 3D np.ndarray encoding the position of each atom.
        """
        if self.__is_bohr is True:
            return [x * bohr_to_angstrom for x in self.__coordinates]
        else:
            return self.__coordinates

    @property
    def coordinates_borh(self) -> List[np.ndarray]:
        """
        The list of coordinates in atomic units (Bohr) encoding the position of
        each atom in the molecule.

        Returns
        -------
        List[np.ndarray]
            The list of 3D np.ndarray encoding the position of each atom.
        """
        if self.__is_bohr is True:
            return self.__coordinates
        else:
            return [x / bohr_to_angstrom for x in self.__coordinates]

    @property
    def cube(self) -> np.ndarray:
        """
        The volumetric data encoded in the cube object.

        Returns
        -------
        np.ndarray
            The numpy array containing the value of the cube property for each voxel.
        """
        return self.__cube

    @property
    def max(self) -> float:
        """
        The maximum value encoded in the cube volumetric data.

        Returns
        -------
        float
            The maximum value encoded in the cube.
        """
        value = None
        for y in self.__cube:
            for z in y:
                zmax = max(z)
                value = zmax if value is None else max([value, zmax])

        return value

    @property
    def min(self) -> float:
        """
        The minimum value encoded in the cube volumetric data.

        Returns
        -------
        float
            The minimum value encoded in the cube.
        """
        value = None
        for y in self.__cube:
            for z in y:
                zmin = min(z)
                value = zmin if value is None else min([value, zmin])

        return value
