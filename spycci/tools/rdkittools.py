import math, logging

from copy import deepcopy
from typing import TYPE_CHECKING, List, Union

from spycci.constants import atoms_dict

from rdkit.Chem import rdchem, rdmolops, rdDetermineBonds

if TYPE_CHECKING:
    from spycci.systems import System

logger = logging.getLogger(__name__)


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
    and sets its connectivity based on the one of the `source` object. In the operation `source` and `destination`
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


def _build_mol_from_system(system: "System", use_mulliken: bool = True) -> rdchem.Mol:
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
    atomic_numbers = {a: i for i, a in atoms_dict.items()}
    for atom in system.geometry.atoms:
        rd_atom = rdchem.Atom(atomic_numbers[atom])
        rd_atom.SetNoImplicit(True)
        rwmol.AddAtom(rd_atom)

    # If Mulliken spin populations are available, use them to set the position of radicals
    if use_mulliken is True:

        spin_populations = system.properties.mulliken_spin_populations

        if spin_populations != [] and system.spin > 1:
            logger.info("- Mulliken spin populations available: Using spin populations to help in radical site determination.")

            ordered = [(i, s) for i, s in enumerate(spin_populations)]
            ordered.sort(key=lambda x: x[1], reverse=True)
            
            radicals = [0 for _ in range(system.geometry.atomcount)]
            for i, s in ordered:
                radicals[i] = math.ceil(s)

                if sum(radicals) + 1 > system.spin:
                    radicals[i] -= sum(radicals) + 1 - system.spin
                
                if sum(radicals) + 1 == system.spin:
                    break
            
            logger.info(f"    -> Radicals assignment: {radicals}")

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
            logger.debug(f"- Trying : connectivity assignmet with charge: {newcharge}")
            rdDetermineBonds.DetermineBonds(newmol, charge=newcharge, embedChiral=True, allowChargedFragments=True)
        
        except:
            logger.debug("    -> Connectivity assignment FAILED")
            continue

        else:
            logger.debug("    -> Connectivity assignment SUCCESS")
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
    logger.debug("- Trying: Adjusting connectivity around affected radical sites.")

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
        logger.debug("        * Radical found on aromatic system: KEKULIZING")

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
                logger.debug(f"        * Radical site {i}: changing bond with atom {idx} from TRIPLE to DOUBLE.")
                break
            
            elif bt == rdchem.BondType.DOUBLE:
                bond.SetBondType(rdchem.BondType.SINGLE)
                other.SetFormalCharge(other_charge)
                logger.debug(f"        * Radical site {i}: changing bond with atom {idx} from DOUBLE to SINGLE.")
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



def system_to_mol(system: "System", catch_errors: bool = True) -> rdchem.Mol:
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

    BEWARE that this function is highly experimental and can fail with open-shell systems or non-standard
    valences. The user MUST carefully review the function output.

    Arguments
    ---------
    system: System
        The input `System` object to be converted
    catch_errors: bool
        If set to `True` (default), will not rise an exception if sanitization fails due to non-standard
        valences. If `False` exception is raised.

    Returns
    -------
    rdkit.Chem.Mol
        An RDKit `Mol` object representing the molecule with explicit hydrogens, 
        connectivity, formal charges, and radical electrons (if any).
    """
    # Build `rdchem.Mol` representation of the input system (no connectivity)
    mol = _build_mol_from_system(system)

    # Connectivity assignment using the `DetermineBonds` function
    # --------------------------------------------------------------------------------------------------------
    
    # If system is singlet, try connectivity assignment as is or convert to triplet    
    if system.spin == 1:

        try:
            logger.debug("- System is in singlet state: running connectivity determination as is.")

            rdDetermineBonds.DetermineBonds(mol, charge=system.charge, embedChiral=True, allowChargedFragments=True)

            #Check for carbene sites and warn the user
            for i, s in enumerate(get_radicals(mol)):
                if s == 2:
                    logger.warning(f"{s} unpaired electrons assigned to site {i} in singlet system: converting carbene to singlet")                       
                    carbene_atom = mol.GetAtomWithIdx(i)
                    carbene_atom.SetNumRadicalElectrons(0)

        except:
            logger.debug("    -> Connectivity assignment FAILED")
            
            # Note: Triplet conversion is largely unused due to conversion to charge pair 
            logger.warning("ASSUMING molecule is a di-radical in singlet state: running conversion using TRIPLET state.")
            obj = deepcopy(system)
            obj.spin = 3
            mol = system_to_mol(obj, catch_errors)

        else:
            logger.debug("    -> Connectivity assignment SUCCESS")            
    
    # If system is multiplet, try connectivity assignment using charge shift
    else:
        logger.debug("- System is open-shell: running heuristic connectivity determination by charge shift.")

        # Generate a guess singlet connectivity by charge shifting
        guess = _guess_connectivity_by_charge_shifting(mol, system.charge, system.spin)
                        
        # If no radical was set (with spin populations), let RDKit attempt to find radicals
        if get_total_number_of_radicals(mol) == 0:
            logger.debug("- Success: Radical assignment not found, using RDKit to find radicals.")
            
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
            logger.debug("- Radical assignment FOUND:")
            logger.debug("    -> Checking if system is compatible with direct copy and PROPERTIES sanitization.")

            newmol : rdchem.Mol = copy_connectivity(guess, mol)   
            sanitized_mol = deepcopy(newmol)
            
            rdmolops.SanitizeMol(
                sanitized_mol,
                sanitizeOps=rdmolops.SanitizeFlags.SANITIZE_PROPERTIES,
                catchErrors=catch_errors
            )

            # Check if the charge and spin multiplicity after sanitization are correct
            charge = get_total_charge(sanitized_mol)
            spin = get_total_number_of_radicals(sanitized_mol) + 1
            
            if charge == system.charge and spin == system.spin:
                logger.debug("- Success: Directly adopting singlet connectivity with radical assignment.")
                mol = sanitized_mol
            
            else:
                logger.debug("    -> Checking compatibility with singlet-based connectivity.")

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
                    logger.debug("- Success: Singlet-based connectivity is VALID.")
                    mol = sanitized_mol
                
                # If radicals would be cleared by sanitization, try to adjust the bond order of the radical site
                else:
                    logger.debug("- Failed: Singlet-based connectivity is INVALID.")
                    logger.debug(f"    -> Affected sites: {affected_sites}")
                    logger.debug("- Trying: Adjusting connectivity around affected radical sites.")

                    mol = _adjust_site_connectivity(newmol, guess, affected_sites, system.charge)

                # Sanitize the molecule setting charges and radicals
                rdmolops.SanitizeMol(
                    mol,
                    sanitizeOps=rdmolops.SanitizeFlags.SANITIZE_PROPERTIES | rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS,
                    catchErrors=catch_errors
                )

    _check_mol_consistency(mol, system.charge, system.spin)

    return mol


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