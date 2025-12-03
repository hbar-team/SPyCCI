import logging

import numpy as np

from typing import Tuple

from spycci.systems import System
from spycci.tools.rdkittools import system_to_mol

from rdkit.Chem import rdchem, rdmolops, rdmolfiles

logger = logging.getLogger(__name__)


def determine_connectivity(system: System) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an input system object of type `System` the function determines the connectivity
    using the `sycci.tools.rdkittools.system_to_mol` function and returns the adjacency and
    bond type matrices. These are  N x N square and symmetrical matrix (with N the total number
    of atoms in the geometry) encoding the molecular connectivity and the order/type of the bonds
    connecting each pair of atoms. If the atom i and j are connected, the [i, j] and [j, i] matrix
    elements of the adjacency matrix will be set to 1 else to 0. At the same time, the same matix 
    elements of the bond type matrix will be set to the float value representing the bond order/type 
    (i.e. 1.0 for single, 2.0 for double, 3.0 for triple, 1.5 for aromatic bonds), otherwise zero if
    they are not connected. 
    
    Note that the bond orders represented here are purely topological and do NOT correspond to 
    quantum-chemically derived bond indices (e.g., Wiberg or Mayer bond orders). PLEASE USE QUANTUM
    CHEMICALLY DERIVED BOND ORDERS IF EXACT BONDING SCHEME IS NEEDED

    Arguments
    ---------
    system: System
        The input `System` object.

    Returns
    -------
    np.ndarray
        The adjacency matrix.
    np.ndarray
        The bond type matrix.
    """
    mol = system_to_mol(system)

    adjacency_matrix : np.ndarray = rdmolops.GetAdjacencyMatrix(mol)

    dim = system.geometry.atomcount
    bond_type_matrix = np.zeros((dim, dim), dtype=float)

    bond: rdchem.Bond = None
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        bond_type_matrix[i, j] = order
        bond_type_matrix[j, i] = order
    
    return adjacency_matrix, bond_type_matrix


def save_sdf(system: System, path: str) -> None:
    """
    Given an input system object of type `System` the function determines the connectivity
    and radical sites using the `sycci.tools.rdkittools.system_to_mol` function and saves the molecular 
    representation to an `.sdf` file at the user specified `path`.

    Arguments
    ---------
    system: System
        The input `System` object.
    path : str
        The full path to the output SDF file. If the file already exists, it will be overwritten.
    """
    mol = system_to_mol(system)

    try:
        writer = rdmolfiles.SDWriter(path)
        writer.write(mol)

    except Exception as e:
        msg = f"An error occurred while saving a `.sdf` file for '{system.name}' system: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    finally:
        writer.close()