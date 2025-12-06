import math, logging

from copy import deepcopy
from typing import List, Optional, Dict

from spycci.systems import System
from spycci.core.geometry import MolecularGeometry
from spycci.constants import atomic_numbers

from rdkit.Chem import rdchem, rdmolops, rdDetermineBonds
    

logger = logging.getLogger(__name__)


###################################################################################################################
#                    GENERAL FUNCTIONS DEDICATED TO OPERATE ON `rdkit.Chem.rdchem.Mol` OBJECTS                    #
###################################################################################################################

def get_charges(mol: rdchem.Mol) -> List[int]:
    """
    Given a `rdkit.Chem.rdchem.Mol` object the function returns the list encoding the formal charges
    assigned to each atom of the molecule.

    Arguents
    --------
    mol: rdchem.Mol
        The `rdkit.Chem.rdchem.Mol` object encoding the molecular structure.

    Raises
    ------
    TypeError
        Exception raised if the argument is not of type `rdkit.Chem.rdchem.Mol` or `rdkit.Chem.rdchem.RWMol`.
    
    Returns
    -------
    List[int]
        The list enconding the formal charges assigned to each atom in the molecule.
    """
    if not isinstance(mol, (rdchem.Mol, rdchem.RWMol)):
        raise TypeError("The `mol` argument must be of type `rdkit.Chem.rdchem.Mol` or `rdkit.Chem.rdchem.RWMol`")
    return [atom.GetFormalCharge() for atom in mol.GetAtoms()]


def get_total_charge(mol: rdchem.Mol) -> int:
    """
    Given a `rdkit.Chem.rdchem.Mol` object the function computes the total charge of the molecule as the sum 
    of formal charges on the atoms.

    Arguents
    --------
    mol: rdchem.Mol
        The `rdkit.Chem.rdchem.Mol` object encoding the molecular structure.

    Raises
    ------
    TypeError
        Exception raised if the argument is not of type `rdkit.Chem.rdchem.Mol` or `rdkit.Chem.rdchem.RWMol`.
    
    Returns
    -------
    int
        The total charge of the molecule as the sum of formal charges.
    """
    charges = get_charges(mol)
    return sum(charges)


def get_radicals(mol: rdchem.Mol) -> List[int]:
    """
    Given a `rdkit.Chem.rdchem.Mol` object the function returns the list encoding the number of radical electrons
    assigned to each atom of the molecule.

    Arguents
    --------
    mol: rdchem.Mol
        The `rdkit.Chem.rdchem.Mol` object encoding the molecular structure.

    Raises
    ------
    TypeError
        Exception raised if the argument is not of type `rdkit.Chem.rdchem.Mol` or `rdkit.Chem.rdchem.RWMol`.
    
    Returns
    -------
    List[int]
        The list enconding the number of radical electrons assigned to each atom in the molecule.
    """
    if not isinstance(mol, (rdchem.Mol, rdchem.RWMol)):
        raise TypeError("The `mol` argument must be of type `rdkit.Chem.rdchem.Mol` or `rdkit.Chem.rdchem.RWMol`")
    return [atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()]


def get_total_number_of_radicals(mol: rdchem.Mol) -> int:
    """
    Given a `rdkit.Chem.rdchem.Mol` object the function computes the total number of radical electrons in the molecule.

    Arguents
    --------
    mol: rdchem.Mol
        The `rdkit.Chem.rdchem.Mol` object encoding the molecular structure.

    Raises
    ------
    TypeError
        Exception raised if the argument is not of type `rdkit.Chem.rdchem.Mol` or `rdkit.Chem.rdchem.RWMol`.
    
    Returns
    -------
    int
        The total number of radical electrons.
    """
    radicals = get_radicals(mol)
    return sum(radicals)


def copy_connectivity(
        source: rdchem.Mol,
        destination: rdchem.Mol,
    ) -> rdchem.Mol:
    """
    Given a `source` and `destination` objects, the function creates a copy of the `destination` molecule
    and sets its connectivity based on the one of the `source` one. In the operation `source` and `destination`
    objects are not altered.

    Arguments
    ---------
    source : rdchem.Mol
        The source object from which the connectivity must be copied
    destination: rdchem.Mol
        The destination object to which the connectivity must be copied

    Returns
    -------
    rdchem.Mol
        A copy of the destination molecule in which the connectivity of the source one has been copied.
    """
    # Check that the molecules have the same number of atoms
    if source.GetNumAtoms() != destination.GetNumAtoms():
        raise RuntimeError("Source and destination molecules have different number of atoms. Cannot copy connectivity.")
    
    # Copy the obtained connectivity to a temporary read-write `Mol` object  
    rwdest = rdchem.RWMol(destination)

    # Clear the connectivity originally stored in the destination molecule
    current_bonds : List[rdchem.Bond] = list(rwdest.GetBonds())[::-1]
    for bond in current_bonds:
        rwdest.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # Copy the connectivity of the source molecule to the destination one
    bond: rdchem.Bond = None
    for bond in source.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        rwdest.AddBond(i, j, bond.GetBondType())
    
    return rwdest.GetMol()


def print_mol(mol: rdchem.Mol, connectivity: bool = True) -> None:
    """
    Given an `rdkit.Chem.rdchem.Mol` object, print a breaf summary of atom properties and connectivity.

    Arguments
    ---------
    mol : rdkit.Chem.rdchem.Mol
        The input `Mol` object
    connectivity: bool
        If set to `True` (default) will print a summary of the connectivity of each atom.
    
    Raises
    ------
    TypeError
        Exception raised if the `mol` argument is not of type `rdkit.Chem.rdchem.Mol`.
    """
    if not isinstance(mol, (rdchem.Mol, rdchem.RWMol)):
        raise TypeError(f"The `mol` argument must be of type `rdkit.Chem.rdchem.Mol`. Invalid type {type(mol)} was used.")
    
    atom: rdchem.Atom = None
    print("ATOMS:")
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        spin = atom.GetNumRadicalElectrons()
        impH = atom.GetTotalNumHs(includeNeighbors=True)
        print(f"Atom {idx:2d}: {sym:2s}, formal charge = {charge}, spin={spin}, Hcount = {impH}")
    
    if connectivity is True:
        print()
        print("CONNECTIVITY:")
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            neigh_info = []

            for nbr in atom.GetNeighbors():
                j = nbr.GetIdx()
                bond = mol.GetBondBetweenAtoms(i, j)

                # bond type as string
                btype = str(bond.GetBondType())

                neigh_info.append(f"{j} {btype}")

            neigh_str = ", ".join(neigh_info) if neigh_info else "—"
            print(f"{i:2d} to: {neigh_str}")

    print("\n")


###################################################################################################################
#  FUNCTIONS DEDICATED TO THE CONVERSION OF A `spycci.systems.System` OBJECT TO A `rdkit.Chem.rdchem.Mol` OBJECT  #
###################################################################################################################

def system_to_mol(
    system: System,
    metal_ox_states: Optional[Dict[int, int]] = None,
    ligand_spin : int = 1,
    catch_errors: bool = True,
) -> rdchem.Mol:
    """
    Given a `System` object, the function generates an `rdkit.Chem.Mol` object from the stored molecular
    geometry, system charge, and spin. The function is based on RDKit and creates a `Mol` object by directly
    converting the stored system geometry in memory through an intermediate `RWMol` read-write molecule object.
    The connectivity of the molecule is automatically assigned using a workflow based on the `DetermineBonds`
    function from the `rdkit.Chem.rdDetermineBonds` module. The conversion process is higly heuristical and
    has been designed to patch some of the limitations of the `DetermineBonds` function in the case of open-shell
    systems. In these cases, bond determination may temporarily adjust the total charge of the system by adding
    or removing electrons (charge-shifting) to create a hypothetical singlet (closed-shell) configuration. The
    obtained connectivity is then copied back to the original molecule, and, for radical systems, radical electrons
    and formal charges are assigned and sanitized using the `SANITIZE_PROPERTIES` and `SANITIZE_FINDRADICALS` 
    options. Implicit hydrogens are not added by default, so radical sites and hydrogen counts are explicit.
    If Mulliken spin populations are available within the system properties, these are automatically used to help
    in the connectivity determination. When spin populations are available, radical sites are set at the beginning
    of the connectivity assignent procedure and are enforced, by adjusting bond orders, when copying the 
    singlet connectivity generated by the charge-shifting approach.

    The case of systems containing metals is particularly problematic and tipically not well handled by the
    standard connectivity determination workflow based on RDKit. To extend the use of the function to metal
    containing systems, the user can provide a dictionary of oxidation states for the metals and a spin multiplicity
    for the ligand backbone. When doing so, the function separates the metal atoms from the organic backbone
    and generates an RDKit `Mol` for the ligand (organic) portion. The metals are then reinserted into the
    molecule with the specified formal charges, ensuring that the final atom ordering matches the original System.
    This approach avoids potential failures of RDKit's `DetermineBonds` function on metal atoms and preserves
    the 3D coordinates of all atoms.

    BEWARE that this function is highly experimental and can fail with open-shell systems or non-standard
    valences. The user MUST carefully review the function output.

    Arguments
    ---------
    system: System
        The input `System` object to be converted
    metal_ox_states : Dict[int, int]
        A dictionary mapping the indices of metal atoms in `system` to their formal oxidation states. 
        Example: {1: 2, 5: 3} where keys are system atom indices and values are formal charges.
    ligand_spin : int
        The spin multiplicity to be assigned to the organic backbone (ligand) after removing metals.
    catch_errors: bool
        If set to `True` (default), will not rise an exception if sanitization fails due to non-standard
        valences. If `False` exception is raised.

    Returns
    -------
    rdkit.Chem.Mol
        An RDKit `Mol` object representing the molecule with explicit hydrogens, 
        connectivity, formal charges, and radical electrons (if any).
    """
    # HANDLE METALS IN THE INPUT STRUCTURE
    # ----------------------------------------------------------------------------------------------------------------
    # Check if metals are present in the system
    metals_detected = False
    if any([atomic_numbers[a] in RDKIT_METALS for a in system.geometry.atoms]):
        metals_detected = True

    # If metals are detected and the user provided a dictionary of oxidation states directly apply them
    if metals_detected is True and metal_ox_states is not None:
        logger.info("Metals detected with assigned oxidation state: Running RDKit on the organic backbone.")
        mol = _process_metals_directly(system, metal_ox_states, ligand_spin=ligand_spin, catch_errors=catch_errors)
        return mol
    
    # If metals are detected issue a warning to the user and try using the standard routine
    elif metals_detected is True:
        logger.warning("Metals detected without informations about the metal oxidation state: Running RDKit anyways.")       

    # START THE CONNECTIVITY ASSIGNMENT ALGORITHM BASED ON RDKIT
    # ----------------------------------------------------------------------------------------------------------------
    
    # Build a `rdchem.Mol` representation of the input system without connectivity
    mol = _build_mol_from_system(system)
    
    # If system is singlet, try connectivity assignment using the `DetermineBonds` function
    if system.spin == 1:

        try:
            logger.info("- System is in singlet state: running connectivity determination as is.")

            rdDetermineBonds.DetermineBonds(mol, charge=system.charge, embedChiral=True, allowChargedFragments=True)

            # Check for carbene sites and warn the user (assumption: setting 0 radical electrons for singlet carbenes)
            for i, s in enumerate(get_radicals(mol)):
                if s == 2:
                    logger.warning(f"{s} unpaired electrons assigned to site {i} in singlet system: converting carbene to singlet")                       
                    carbene_atom = mol.GetAtomWithIdx(i)
                    carbene_atom.SetNumRadicalElectrons(0)

        # If standard conversion fails try running the conversion using an hypotetical TRIPLET state (Assuming di-radical)
        # Note: Triplet conversion is largely unused due to conversion to charge pair 
        except:
            logger.info("    -> Connectivity assignment FAILED")
            logger.warning("ASSUMING molecule is a di-radical in singlet state: running conversion using TRIPLET state.")
            obj = deepcopy(system)
            obj.spin = 3
            mol = system_to_mol(obj, catch_errors)

        else:
            logger.info("    -> Connectivity assignment SUCCESS")            
    
    # If system is multiplet (open-shell), try connectivity assignment using charge shift
    else:
        logger.info("- System is open-shell: running heuristic connectivity determination by charge shift.")

        # Generate a guess singlet connectivity by charge shifting
        guess = _guess_connectivity_by_charge_shifting(mol, system.charge, system.spin)
                        
        # If no radical was set (with spin populations), let RDKit attempt to find radicals
        if get_total_number_of_radicals(mol) == 0:
            logger.info("- Success: Radical assignment not found, using RDKit to find radicals.")
            
            # Directly copy back the charge shifted connectivity to the original `Mol` object
            mol : rdchem.Mol = copy_connectivity(guess, mol)

            # Sanitize the molecule setting charges and radicals
            rdmolops.SanitizeMol(
                mol,
                sanitizeOps=rdmolops.SanitizeFlags.SANITIZE_PROPERTIES | rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS,
                catchErrors=catch_errors
            )

        # If radicals were set (with spin populations) check if they are compatible with singlet connectivity
        else:
            logger.info("- Radical assignment FOUND:")
            logger.info("    -> Checking if system is compatible with direct copy and PROPERTIES sanitization.")
            
            # Copy the singlet connectivity and check if charge and spin are correct after properties sanitization
            newmol : rdchem.Mol = copy_connectivity(guess, mol)   
            
            # Create a copy of the temporary read-write `Mol` object and sanitize it.
            sanitized_mol = deepcopy(newmol)
            
            rdmolops.SanitizeMol(
                sanitized_mol,
                sanitizeOps=rdmolops.SanitizeFlags.SANITIZE_PROPERTIES,
                catchErrors=catch_errors
            )

            charge = get_total_charge(sanitized_mol)
            spin = get_total_number_of_radicals(sanitized_mol) + 1
            
            # If yes: simply return the sanitized `Mol` object
            if charge == system.charge and spin == system.spin:
                logger.info("- Success: Directly adopting singlet connectivity with radical assignment.")
                mol = sanitized_mol
            
            # If not: sanitize using also the FINDRADICALS option and check if the found radicals are compatible with
            #         the singlet connectivity guess
            else:
                logger.info("    -> Checking compatibility with singlet-based connectivity.")

                # Create a copy of the temporary read-write `Mol` object and sanitize it.
                sanitized_mol = deepcopy(newmol)
                
                rdmolops.SanitizeMol(
                    sanitized_mol,
                    sanitizeOps=rdmolops.SanitizeFlags.SANITIZE_PROPERTIES | rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS,
                    catchErrors=catch_errors
                )

                # Check if the originally set radicals have been removed and which are affected
                radicals = get_radicals(mol)
                affected_radicals = [False for _ in radicals]
                for i, nrad in enumerate(radicals):
                    sanitized_atom = sanitized_mol.GetAtomWithIdx(i)
                    sanitized_nrad = sanitized_atom.GetNumRadicalElectrons()
                    if sanitized_nrad < nrad:
                        affected_radicals[i] = True
                
                affected_sites = [i for i, b in enumerate(affected_radicals) if b is True]
                
                # If the radicals have been maintained, simply copy the molecule
                if affected_sites == []:
                    logger.info("- Success: Singlet-based connectivity is VALID.")
                    mol = sanitized_mol
                
                # If radicals would be cleared by sanitization, try to adjust the bond order of the radical site
                else:
                    logger.info("- Failed: Singlet-based connectivity is INVALID.")
                    logger.info(f"    -> Affected sites: {affected_sites}")
                    logger.info("- Trying: Adjusting connectivity around affected radical sites.")

                    mol = _adjust_site_connectivity(newmol, guess, affected_sites, system.charge)

                # Sanitize the molecule setting charges and radicals
                rdmolops.SanitizeMol(
                    mol,
                    sanitizeOps=rdmolops.SanitizeFlags.SANITIZE_PROPERTIES | rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS,
                    catchErrors=catch_errors
                )

    _check_mol_consistency(mol, system.charge, system.spin)

    return mol


# *****************************************************************************************************************
# *                                               HELPER FUNCTIONS                                                *
# *****************************************************************************************************************

# List of "METALS" that may cause problems with the RDKit `DetermineBonds` function
RDKIT_METALS = {
    # s-block
    3,  4, 11, 12, 19, 20, 37, 38, 55, 56,

    # p-block
    13, 31, 49, 50, 81, 82, 83, 32, 33, 52,
    
    # d-block
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    
    # f-block
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
}


def _process_metals_directly(
    system: System,
    metal_ox_states: Dict[int, int],
    ligand_spin: int = 1,
    catch_errors: bool = True
) -> rdchem.Mol:
    """
    Process a molecular `System` containing metals by separating the metal atoms from the organic 
    backbone, generating a RDKit `Mol` for the "organic" part (ligand), and then reinserting the metals 
    with their specified oxidation states. The function ensures that the final RDKit Mol has the
    same atom order as the original `System` and preserves the 3D coordinates of all atoms.

    Parameters
    ----------
    system : System
        The input molecular system containing both metals and organic atoms. 
    metal_ox_states : Dict[int, int]
        A dictionary mapping the indices of metal atoms in `system` to their formal oxidation states. 
        Example: {1: 2, 5: 3} where keys are system atom indices and values are formal charges.
    ligand_spin : int
        The spin multiplicity to assign to the organic backbone (ligand) after removing metals.
    catch_errors : bool
        Whether to catch errors during the conversion of the organic backbone to RDKit Mol.

    Returns
    -------
    rdchem.Mol
        An RDKit `Mol` object containing all atoms.
    """
    # Detect all the metals in the molecule
    detected_metals = []
    for i, atom in enumerate(system.geometry.atoms):
        if atomic_numbers[atom] in RDKIT_METALS:
            detected_metals.append(i)

    # Check if the user provided oxidation states list matches the detected metal list
    if len(detected_metals) > len(metal_ox_states):
        raise RuntimeError("The number of detected metals is larger then the list of provided oxidation states.")

    for i in detected_metals:
        if i not in metal_ox_states.keys():
            raise RuntimeError("Mismatch between the provided oxidation states and the list of detected metals.")
    
    # Define an indices list to keep track of the atom sequence
    indices = []

    # Define a `System` in which the metal atoms have been removed and the charge is
    # lowered to take into account the charge brought to the system by the metal
    geometry = MolecularGeometry()
    atoms = system.geometry.atoms
    coordinates = system.geometry.coordinates
    for i, (atom, coordinates) in enumerate(zip(atoms, coordinates)):

        if i in detected_metals:
            continue

        geometry.append(atom, coordinates)
        indices.append(i)
    
    new_charge = system.charge - sum(metal_ox_states.values())
    new_system = System(system.name, geometry, charge=new_charge, spin=ligand_spin)

    # Run system to molecule conversion on the organic backbone
    logger.info(f"-> Running ligand conversion with charge {new_charge} and spin multiplicity {ligand_spin}.")
    ligand = system_to_mol(new_system, catch_errors=catch_errors)

    # Create a RWMol representation of the molecule and add the missing metals with their formal charges
    rwmol = rdchem.RWMol(ligand)

    for i in detected_metals:

        indices.append(i)

        # Add the atom to the molecule
        symbol = system.geometry.atoms[i]
        rd_atom = rdchem.Atom(atomic_numbers[symbol])
        rd_atom.SetNoImplicit(True)
        rd_atom.SetFormalCharge(metal_ox_states[i])
        idx = rwmol.AddAtom(rd_atom)

        # Update the coordinate of the atom in the conformer
        conf = rwmol.GetConformer(id=0)
        coordinates = system.geometry.coordinates[i]
        conf.SetAtomPosition(idx, coordinates)

    # Renumber the atoms to ensure the sequence is the same as the one of the original system
    permutations = [indices.index(i) for i in range(system.geometry.atomcount)]
    mol = rdmolops.RenumberAtoms(rwmol, permutations)
    
    return mol


def _build_mol_from_system(system: System, use_mulliken: bool = True) -> rdchem.Mol:
    """
    Given a `System` object, the function generates an `rdkit.Chem.Mol` object encoding the molecular structure
    of the given system. A single conformer is created encoding the system geometry. If Mulliken spin populations
    are available, these are used to assign localized radical electrons. To do so, the spin polulation list is
    sorted and radical electrons are set until the right spin multiplicity has been obtained. Beware that no
    connectivity deterimination nor sanitation is carried out by the function.

    Arguments
    ---------
    system: System
        The input `System` object to be converted.
    use_mulliken: bool
        If set to `True` (default) will use the Mulliken spin populations (if available) to set localized
        radical electrons.
    
    Returns
    -------
    rdkit.Chem.Mol
        An RDKit `Mol` object representing the `System` object.
    """
    logger.info(f"Generating RDKit Mol object from '{system.name}' system (charge: {system.charge}, spin: {system.spin})")

    # Create an empty instance of a read-write molecule object
    rwmol = rdchem.RWMol()

    # Initialize the atom list of the `Mol` object with the system `atoms` list 
    for atom in system.geometry.atoms:
        rd_atom = rdchem.Atom(atomic_numbers[atom])
        rd_atom.SetNoImplicit(True)
        rwmol.AddAtom(rd_atom)

    # If Mulliken spin populations are available, use them to set the position of radicals
    if use_mulliken is True:

        spin_populations = system.properties.mulliken_spin_populations

        if spin_populations != [] and system.spin > 1:
            logger.info("- Mulliken spin populations available: Using spin populations to help in radical site determination.")
            
            # Create a list of tuples encoding atom index and spin poupulation and sort them
            # according to decreasing spin population values
            ordered = [(i, s) for i, s in enumerate(spin_populations)]
            ordered.sort(key=lambda x: x[1], reverse=True)
            
            # Define a list of radicals and set them according to the ceiling value of the spin population of the site
            radicals = [0 for _ in range(system.geometry.atomcount)]
            for i, s in ordered:
                radicals[i] = math.ceil(s)

                # If the assigned radicals exceeds spin multipicity (unlikely) correct the last assignment
                if sum(radicals) + 1 > system.spin:
                    radicals[i] -= sum(radicals) + 1 - system.spin
                
                # If the right spin multiplicity is obtained break
                if sum(radicals) + 1 == system.spin:
                    break
            
            logger.info(f"    -> Radicals assignment: {radicals}")

            # Apply the radical list to the atoms of the `Mol` object
            for i, s in enumerate(radicals):
                if s>0:
                    atom = rwmol.GetAtomWithIdx(i)
                    atom.SetNumRadicalElectrons(s)

    # Convert the read-write molecule object to a standard `Mol` object
    mol = rwmol.GetMol()

    # Assign a conformer to the `Mol` object holding the coordinates of the atoms in the system
    conf = rdchem.Conformer(system.geometry.atomcount)
    for i, coords in enumerate(system.geometry.coordinates):
        conf.SetAtomPosition(i, coords)

    mol.AddConformer(conf, assignId=True)

    return mol


def _guess_connectivity_by_charge_shifting(mol: rdchem.Mol, charge: int, spin: int) -> rdchem.Mol:
    """
    Given an input `rdchem.Mol` object associated with defined charge and spin values, the function try to predict
    a guess connectivity by charge shifting. During the process the charge of the molecule can be altered to generate
    hypotetical closed shell system configurations.

    Arguments
    ---------
    mol: rdchem.Mol
        The RDKit `Mol` object for which a guess connectivity needs to be generated.
    charge: int
        The expected charge of the system.
    spin: int
        The expected spin multiplicity of the system.

    Raises
    ------
    RuntimeError
        Exception raised if the connectivity generation by charge shifting is unsuccesful.

    Returns
    -------
    mol: rdchem.Mol
        The RDKit `Mol` object encoding the guess connectivity
    """
    newmol: rdchem.Mol = None
    
    for charge_shift in [0, spin - 1, 1 - spin]:
        
        try:
            newmol = deepcopy(mol)
            newcharge = charge + charge_shift
            logger.info(f"- Trying : connectivity assignmet with charge: {newcharge}")
            rdDetermineBonds.DetermineBonds(newmol, charge=newcharge, embedChiral=True, allowChargedFragments=True)
        
        except:
            logger.info("    -> Connectivity assignment FAILED")
            continue

        else:
            logger.info("    -> Connectivity assignment SUCCESS")
            break
    
    else:
        msg = f"- Failure: Connectivity assignment failed for open-shell system"
        logger.error(msg)
        raise RuntimeError(msg)

    return newmol


def _adjust_site_connectivity(mol: rdchem.Mol, guess: rdchem.Mol, affected_sites: List[int], charge: int) -> rdchem.Mol:
    """
    Given an input `mol` object encoding atoms with assigned properties (i.e. formal charges and radicals) and
    failing to be correctly sanitized by RDKit, the functions try to adjust the `guess` singlet connectivity to
    account for radical formation due to multiple bonds ionization/electron addition. The function adjust connectivity
    around the `affected_sites` trying to match the real system `charge`.

    Arguments
    --------
    mol : rdchem.Mol
        The `rdchem.Mol` object encoding the assigned properties (i.e. formal charges and radicals).
    guess : rdchem.Mol
        The `rdchem.Mol` object encoding the singlet connectivity guess failing the properties and radical sanitization
        operation when adopted in an open-shell system.
    affected_sites : List[int]
        The list encoding the index of the sites the connectivity of which must be adjusted.
    charge: int
        The charge of the system
    
    Returns
    -------
    rdchem.Mol
        The `rdchem.Mol` object with adjusted molecular connectivity.
    """
    logger.info("- Trying: Adjusting connectivity around affected radical sites.")

    # Create a writable copy of the original (open-shell) molecule to adjust the connectivity
    rwmol = rdchem.RWMol(mol)

    # Check if one of the radicals is located on an aromatic ring
    radical_on_aromatic = False
    for atom in mol.GetAtoms():

        if atom.GetNumRadicalElectrons() == 0:
            continue

        if any([bond.GetIsAromatic() for bond in atom.GetBonds()]):
            radical_on_aromatic = True
        
    # If one of the radicals is set on an aromatic system kekulize the singlet connectivity guess
    if radical_on_aromatic is True:
        logger.info("        * Radical found on aromatic system: KEKULIZING")

        # Copy the singlet guess geometry and Kekulize it
        newmol = deepcopy(guess)
        rdmolops.SanitizeMol(newmol, sanitizeOps=rdmolops.SanitizeFlags.SANITIZE_KEKULIZE)
 
        # Copy the new kekulized connectivity to the temporary read-write `Mol` object
        rwmol = rdchem.RWMol(copy_connectivity(newmol, mol))

    # Adjust the connectivity of the radical site by breaking multiple bonds and setting charges
    for i in affected_sites:

        # Check the current charge of the system
        current_charge = get_total_charge(rwmol)
        
        atom: rdchem.Atom = rwmol.GetAtomWithIdx(i)
        
        bond : rdchem.Bond = None
        for bond in atom.GetBonds():
            
            # Get the bond type
            bt = bond.GetBondType()

            if bt == rdchem.BondType.SINGLE:
                continue

            elif bt == rdchem.BondType.AROMATIC:
                msg = f"Bond type AROMATIC detected on site ({i}) after kekulization."
                logger.error(msg)
                raise RuntimeError(msg)

            # Identify the partner atom (other) and update its formal charge according to the total charge of the system
            idx = bond.GetBeginAtomIdx() if bond.GetBeginAtomIdx() != i else bond.GetEndAtomIdx()
            other = rwmol.GetAtomWithIdx(idx)
            other_charge = other.GetFormalCharge()
            other_charge += 1 if current_charge < charge else -1

            if bt == rdchem.BondType.TRIPLE:
                bond.SetBondType(rdchem.BondType.DOUBLE)
                other.SetFormalCharge(other_charge)
                logger.info(f"        * Radical site {i}: changing bond with atom {idx} from TRIPLE to DOUBLE.")
                break
            
            elif bt == rdchem.BondType.DOUBLE:
                bond.SetBondType(rdchem.BondType.SINGLE)
                other.SetFormalCharge(other_charge)
                logger.info(f"        * Radical site {i}: changing bond with atom {idx} from DOUBLE to SINGLE.")
                break
            
    return rwmol.GetMol()
       

def _check_mol_consistency(mol: rdchem.Mol, charge: int, spin: int) -> None:
    """
    Given ad `rdchem.Mol` object, the function checks wheter the results matches expected charge and spin
    target values. If an odd number of unpaired electrons is found in singlet systems, no radical if found
    in open shell systems or the wrong spin multiplicity / charge is assigned, a `RuntimeError` excetpion
    is raised.

    Arguments
    ---------
    mol: rdchem.Mol
        The RDKit `Mol` object that needs to be checked.
    charge: int
        The expected charge of the system.
    spin: int
        The expected spin multiplicity of the system.

    Raises
    ------
    RuntimeError
        Exception raised if the `Mol` object is not compatible with charge and spin target values.
    """
    radicals = get_radicals(mol)
    num_radicals = get_total_number_of_radicals(mol)
    
    if spin == 1:

        if num_radicals % 2 != 0:
            msg = "An odd number of radical electrons has been assigned in singlet system."
            logger.error(msg)
            raise RuntimeError(msg)
        
        for i, s in enumerate(radicals):
            if s > 0:
                logger.info(f"Non-zero ({s}) unpaired electron assigned to site {i} in a singlet system")
    
    else:

        if all([s == 0 for s in radicals]):
            msg = "None of the atoms in the generated `Mol` object have radical electrons even if the system is open-shell."
            logger.error(msg)
            raise RuntimeError(msg)
        
        elif num_radicals != spin-1:
            msg = f"The sum of unpaired electrons ({num_radicals}) is different from the one ({spin-1}) expected from spin multiplicity."
            logger.error(msg)
            raise RuntimeError(msg)
    
    # Check the sum of formal charges warn the user if something looks strange
    if get_total_charge(mol) != charge:
        msg = "The sum of formal charges does not match the total charge of the system."
        logger.error(msg)
        raise RuntimeError(msg)