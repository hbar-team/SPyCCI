import logging

import numpy as np

from typing import Tuple, List

from spycci.systems import System
from spycci.constants import atoms_dict
from spycci.tools.rdkittools import system_to_mol, get_total_charge, get_total_number_of_radicals

from rdkit.Chem import rdchem, rdmolops, rdmolfiles

logger = logging.getLogger(__name__)


class ChemInfo:
    """
    The `ChemInfo` represents a simple class designed to wrap a `System` object extending its application to the field
    of cheminformatic. While a `System` object represents a container for atomic coordinates and "exact" properties
    derived from computational chemistry calculations, the `ChemInfo` represent a broader container designed to give to
    the user a set of tools to explore molecular connectivity, structural properties and cheminformatic descriptors based
    on heuristic rules or data not derived from computational chemistry calculations. The core of the class is based on
    the RDKit library that is tasked with the connectivity determination. To create an instance of the `ChemInfo` an 
    instance of a `System` class must be provided. The given system object is stored (deepcopied) in the private class
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


    def get_connectivity(self) -> Tuple[np.ndarray, np.ndarray]:
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
    

    def search_SMARTS(self, smarts: str) -> Tuple[Tuple[int]]:
        """
        Given a SMARTS string, query the molecular representation searching for matches. Once found all the matches,
        the function returns a tuple of tuples encoding the indices of the molecule’s atoms that match the query.

        Arguments
        ---------
        smarts: str
            The SMARTS string to be used in the substructure search.
        
        Retruns
        -------
        Tuple[Tuple[int]]
            The tuple of tuples encoding the indices of the molecule’s atoms that match the query
        """
        query = rdmolfiles.MolFromSmarts(smarts)
        results = self.__mol.GetSubstructMatches(query)
        return results
    

    def locate_hydrogen_bonds(
        self,
        donor_list: List[str] = ['N', 'O', 'F'],
        acceptor_list: List[str] = ['N', 'O', 'F'],
        hbond_max_distance: float = 2.5,
        hbond_min_angle: float = 120.,
    ) -> List[List[int]]:
        """
        The function locates all possible hydrogen bonds within the system. The function queries the molecular
        structure searching for hydrogen donors and acceptors and, if found, checks whether an hydrogen bond is possible.

        Arguments
        ---------
        donor_list: List[str]
            The list of atoms that should be considered as possible hydrogen donors. (default: ['N', 'O', 'F'])
        acceptor_list: List[str]
            The list of atoms that should be considered as possible hydrogen acceptors. (default: ['N', 'O', 'F'])
        hbond_max_distance: float
            The maximum distance (in Angstrom) between the hydrogen and the acceptor atom. (default: 2.5)
        hbond_min_angle: float
            The minimum angle D-H-A (in degrees) formed by the donor (D), hydrogen (H) and acceptror (A). (default: 120°)
        
        Returns
        -------
        List[List[int]]
            The list of lists encoding the detected hydrogen bonds. Each inner list contains the index of the hydrogen and
            that of the acceptor atom.
        """
        # Generate SMARTS string to search for donor and acceptor atoms
        atomic_numbers = {a: i for i, a in atoms_dict.items()}
        donor_smarts = "[!H0;" + ",".join([f"#{atomic_numbers[s]}" for s in donor_list]) + "]"
        acceptor_smarts = "[" + ",".join([f"#{atomic_numbers[s]}" for s in acceptor_list]) + "]"

        # Query the molecule for donor and acceptor atoms
        donors = self.search_SMARTS(donor_smarts)
        acceptors = self.search_SMARTS(acceptor_smarts)

        # Iterate on the donor atoms, find the connected hydrogens and probe all possible acceptor
        hbonds = []
        for d in [idx[0] for idx in donors]:

            donor_atom = self.__mol.GetAtomWithIdx(d)
            neighbors : List[rdchem.Atom] = donor_atom.GetNeighbors()

            for neighbor in neighbors:
                
                if neighbor.GetAtomicNum() != 1:
                    continue
                
                h = neighbor.GetIdx()

                for a in [idx[0] for idx in acceptors]:

                    if a == d:
                        continue

                    distance = self.__system.geometry.distance(h, a)
                    if distance > hbond_max_distance:
                        continue

                    angle = self.__system.geometry.angle(d, h, a)
                    if angle < (np.pi/180.)*hbond_min_angle:
                        continue

                    hbonds.append([h, a])
        
        return hbonds