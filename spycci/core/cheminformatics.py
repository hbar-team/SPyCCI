import logging

import numpy as np

from typing import Tuple

from spycci.systems import System
from spycci.tools.rdkittools import system_to_mol, get_total_charge, get_total_number_of_radicals

from rdkit.Chem import rdchem, rdmolops, rdmolfiles

logger = logging.getLogger(__name__)


class CheminformaticWrapper:
    """
    The `CheminformaticWrapper` represents a simple class designed to wrap a `System` object extending its application to
    the field of cheminformatic. While a `System` object represents a container for atomic coordinates and "exact" properties
    derived from computational chemistry calculations, the `CheminformaticWrapper` represent a broader container designed to
    give to the user a set of tools to explore molecular connectivity, structural properties and cheminformatic descriptors
    based on heuristic rules or data not derived from computational chemistry calculations. The core of the class is based on
    the RDKit library that is tasked with the connectivity determination. To create an instance of the `CheminformaticWrapper`
    an instance of a `System` class must be provided. The given system object is stored (deepcopied) in the private class
    attributes and connectivity is determined using the built-in `spycci.tools.rdkittools.system_to_mol` function.

    Arguments
    ---------
    system: System
        The input `System` object encoding the structure of interest.
    """

    def __init__(self, system: System) -> None:
        
        if not isinstance(system, System):
            raise TypeError(f"The `system` argument must be of type `System`. Type {type(system)} was given.")
        
        self.__system : System = system
        self.__mol : rdchem.Mol = system_to_mol(system)


    def determine_connectivity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the  adjacency and bond type matrices. These are  N x N square and symmetrical matrix 
        (with N the total number of atoms in the geometry) encoding the molecular connectivity and the
        order/type of the bonds connecting each pair of atoms. If the atom i and j are connected, the
        [i, j] and [j, i] matrix elements of the adjacency matrix will be set to 1 else to 0. At the same
        time, the same matix elements of the bond type matrix will be set to the float value representing
        the bond order/type (i.e. 1.0 for single, 2.0 for double, 3.0 for triple, 1.5 for aromatic bonds),
        otherwise zero if they are not connected. 
        
        Note that the bond orders represented here are purely topological and do NOT correspond to 
        quantum-chemically derived bond orders (e.g., Wiberg or Mayer bond orders). PLEASE USE QUANTUM
        CHEMICALLY DERIVED BOND ORDERS IF EXACT BONDING SCHEME IS NEEDED

        Returns
        -------
        np.ndarray
            The adjacency matrix.
        np.ndarray
            The bond type matrix.
        """
        adjacency_matrix : np.ndarray = rdmolops.GetAdjacencyMatrix(self.__mol)

        dim = self.__system.geometry.atomcount
        bond_type_matrix = np.zeros((dim, dim), dtype=float)

        bond: rdchem.Bond = None
        for bond in self.__mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            order = bond.GetBondTypeAsDouble()
            bond_type_matrix[i, j] = order
            bond_type_matrix[j, i] = order
        
        return adjacency_matrix, bond_type_matrix


    def save_sdf(self, path: str) -> None:
        """
        Saves the molecular representation to an `.sdf` file at the user specified `path`.

        Arguments
        ---------
        path : str
            The full path to the output (`.sdf`) file. If the file already exists, it will be overwritten.
        """
        try:
            writer = rdmolfiles.SDWriter(path)
            writer.write(self.__mol)

        except Exception as e:
            msg = f"An error occurred while saving a `.sdf` file for '{self.__system.name}' system: {e}"
            logger.error(msg)
            raise RuntimeError(msg)

        finally:
            writer.close()