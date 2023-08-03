from __future__ import annotations

from os.path import isfile
from typing import List
from copy import deepcopy

import numpy as np


class Cube:
    """
    Simple Cube class allowing the loading, saving and manipulation of cube files.
    """

    def __init__(self) -> None:
        self.__natoms: int = None  # Number of atoms in the molecule
        self.__origin: np.ndarray = None  # Position of the volumetric data
        self.__nvoxels: List[int] = []  # Number of voxels for each dimension
        self.__axis: List[np.ndarray] = []  # List of axis vectors
        self.__atomic_numbers: List[int] = []  # List of atomic numbers of the atoms
        self.__atomic_charges: List[float] = []  # List of atomic "charges" of the atoms
        self.__coordinates: List[np.ndarray] = []  # List of coordinate vectors of each atom
        self.__cube: np.ndarray = None  # The volumetric data in cube format

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
            obj.__natoms = int(data[0])
            obj.__origin = np.array([float(value) for value in data[1::]])

            # Read the number of voxel along each axis and the axis vector
            for _ in range(3):
                data = file.readline().split()
                obj.__nvoxels.append(int(data[0]))
                obj.__axis.append(np.array([float(value) for value in data[1::]]))

            # Read the section containing the atomic positions
            for _ in range(obj.__natoms):
                data = file.readline().split()
                obj.__atomic_numbers.append(int(data[0]))
                obj.__atomic_charges.append(float(data[1]))
                obj.__coordinates.append(np.array([float(value) for value in data[2::]]))

            # Read the whole cube section
            buffer = []
            for line in file.readlines():
                sline = line.strip("\n").split()
                for element in sline:
                    buffer.append(float(element))
            
            x = []
            for i in range(obj.__nvoxels[0]):
                y = []
                for j in range(obj.__nvoxels[1]):
                    z = []
                    for k in range(obj.__nvoxels[2]):
                        index = i*obj.__nvoxels[1]*obj.__nvoxels[2] + j*obj.__nvoxels[2] + k
                        z.append(buffer[index])
                    y.append(np.array(z))
                x.append(np.array(y))

            obj.__cube = np.array(x)

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
            file.write(f"\t{self.__natoms}")
            for value in self.__origin:
                file.write(f"\t{value:.6e}")
            file.write("\n")

            # Write the number of voxel along each axis and the axis vector
            for i in range(3):
                file.write(f"\t{self.__nvoxels[i]}")
                for value in self.__axis[i]:
                    file.write(f"\t{value:.6e}")
                file.write("\n")

            # Write the section containing the atomic positions
            for i in range(self.__natoms):
                file.write(f"\t{self.__atomic_numbers[i]}")
                file.write(f"\t{self.__atomic_charges[i]:.6e}")
                for value in self.__coordinates[i]:
                    file.write(f"\t{value:.6e}")
                file.write("\n")

            # Read the whole cube section
            for i in range(self.__nvoxels[0]):
                for j in range(self.__nvoxels[1]):
                    for k in range(self.__nvoxels[2]):
                        file.write(f"\t{self.__cube[i][j][k]:.6e}")
                        if k % 6 == 5:
                            file.write("\n")
                    file.write("\n")

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
                self.__natoms == other.__natoms,
                self.__nvoxels == other.__nvoxels,
                np.allclose(self.__origin, other.__origin, rtol=rtol),
                np.allclose(self.__axis, other.__axis, rtol=rtol),
                np.allclose(self.__coordinates, other.__coordinates, rtol=rtol),
            ]
        ):
            raise ValueError(
                "Cannot perform the operation, the cubes object are not compatible"
            )

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
    def cube(self) -> np.ndarray:
        """
        The volumetric data encoded in the cube object

        Returns
        -------
        np.ndarray
            The numpy array containing the value of the cube property for each voxel
        """
        return self.__cube

    @property
    def charges(self) -> List[float]:
        """
        The charges associated to each atom in the molecule

        Returns
        -------
        List[float]
            The list of charges associated to each atom in the cube
        """
        return self.__atomic_charges
    
    @charges.setter
    def charges(self, new_charges: List[float]) -> None:
        if len(new_charges) != self.__natoms:
            raise ValueError(f"Cannot set charges, {len(new_charges)} values given for a {self.__natoms} atom molecule")
        self.__atomic_charges = new_charges