import pytest

from os.path import abspath, dirname, join

from spycci.systems import System
from spycci.tools.rdkittools import system_to_mol, print_mol

from spycci.core.base import Engine

from rdkit.Chem import rdchem

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))
XYZPATH = join(TEST_DIR, "utils/xyz_examples")

################################################################################################
#                          TEST system_to_mol WITHOUT SPIN POPULAIONS                          #
################################################################################################

def test_system_to_mol_allyl_radical_no_spin():

    xyz_file = f"{XYZPATH}/allyl.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=2)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0, 0, 0, 0, 0, 0]
    expected_nrad      = [0, 0, 0, 0, 1, 0, 0, 0]
    expected_hydrogens = [1, 2, 0, 0, 2, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_allyl_cation_no_spin():

    xyz_file = f"{XYZPATH}/allyl.xyz"
    system = System.from_xyz(xyz_file, charge=1, spin=1)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0, 0, 1, 0, 0, 0]
    expected_nrad      = [0, 0, 0, 0, 0, 0, 0, 0]
    expected_hydrogens = [1, 2, 0, 0, 2, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_allyl_anion_no_spin():

    xyz_file = f"{XYZPATH}/allyl.xyz"
    system = System.from_xyz(xyz_file, charge=-1, spin=1)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0, 0, -1, 0, 0, 0]
    expected_nrad      = [0, 0, 0, 0, 0, 0, 0, 0]
    expected_hydrogens = [1, 2, 0, 0, 2, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"

@pytest.mark.xfail
def test_system_to_mol_aryl_radical_no_spin():

    xyz_file = f"{XYZPATH}/aryl.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=2)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_nrad      = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    expected_hydrogens = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


@pytest.mark.xfail
def test_system_to_mol_aryl_cation_no_spin():

    xyz_file = f"{XYZPATH}/aryl.xyz"
    system = System.from_xyz(xyz_file, charge=1, spin=1)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    print_mol(mol)

    expected_charge    = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    expected_nrad      = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_hydrogens = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_aryl_anion_no_spin():

    xyz_file = f"{XYZPATH}/aryl.xyz"
    system = System.from_xyz(xyz_file, charge=-1, spin=1)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0]
    expected_nrad      = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_hydrogens = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_benzene_no_spin():

    xyz_file = f"{XYZPATH}/benzene.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=1)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_nrad      = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_hydrogens = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


@pytest.mark.xfail
def test_system_to_mol_benzene_radical_cation_no_spin():

    xyz_file = f"{XYZPATH}/benzene.xyz"
    system = System.from_xyz(xyz_file, charge=1, spin=2)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_nrad      = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_hydrogens = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


@pytest.mark.xfail
def test_system_to_mol_benzene_radical_anion_no_spin():

    xyz_file = f"{XYZPATH}/benzene.xyz"
    system = System.from_xyz(xyz_file, charge=-1, spin=2)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_nrad      = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_hydrogens = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_carbene_singlet_no_spin():

    xyz_file = f"{XYZPATH}/carbene.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=1)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0]
    expected_nrad      = [0, 0, 0]
    expected_hydrogens = [2, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_carbene_triplet_no_spin():

    xyz_file = f"{XYZPATH}/carbene.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=3)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0]
    expected_nrad      = [2, 0, 0]
    expected_hydrogens = [2, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_diradical_triplet_no_spin():

    xyz_file = f"{XYZPATH}/diradical.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=3)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_nrad      = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    expected_hydrogens = [2, 2, 0, 0, 2, 0, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_methyl_radical_no_spin():

    xyz_file = f"{XYZPATH}/methyl.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=2)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0, 0]
    expected_nrad      = [1, 0, 0, 0]
    expected_hydrogens = [3, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_methyl_cation_no_spin():

    xyz_file = f"{XYZPATH}/methyl.xyz"
    system = System.from_xyz(xyz_file, charge=1, spin=1)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [1, 0, 0, 0]
    expected_nrad      = [0, 0, 0, 0]
    expected_hydrogens = [3, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_methyl_anion_no_spin():

    xyz_file = f"{XYZPATH}/methyl.xyz"
    system = System.from_xyz(xyz_file, charge=-1, spin=1)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [-1, 0, 0, 0]
    expected_nrad      = [0, 0, 0, 0]
    expected_hydrogens = [3, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"

@pytest.mark.xfail
def test_system_to_mol_nitric_oxide_no_spin():

    xyz_file = f"{XYZPATH}/nitric_oxide.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=2)

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0]
    expected_nrad      = [1, 0, 0]
    expected_hydrogens = [0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


################################################################################################
#                            TEST system_to_mol WITH SPIN POPULAIONS                           #
################################################################################################

def test_system_to_mol_allyl_radical_with_spin():

    xyz_file = f"{XYZPATH}/allyl.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=2)

    spin_populations = [0.0, 0.4996, 0.0, 0.0, 0.5004, 0.0, -0.0, 0.0]
    system.properties.set_mulliken_spin_populations(spin_populations, Engine("dummy"))

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0, 0, 0, 0, 0, 0]
    expected_nrad      = [0, 0, 0, 0, 1, 0, 0, 0]
    expected_hydrogens = [1, 2, 0, 0, 2, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_aryl_radical_with_spin():

    xyz_file = f"{XYZPATH}/aryl.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=2)

    spin_populations = [0.0538, 0.0336, 0.0018, 0.0538, 0.0336, 0.7673, 0.0102, 0.0134, 0.0089, 0.0102, 0.0134]
    system.properties.set_mulliken_spin_populations(spin_populations, Engine("dummy"))

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_nrad      = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    expected_hydrogens = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_benzene_radical_cation_with_spin():

    xyz_file = f"{XYZPATH}/benzene.xyz"
    system = System.from_xyz(xyz_file, charge=1, spin=2)

    spin_populations = [0.1612, 0.1612, 0.1612, 0.1612, 0.1612, 0.1612, 0.0055, 0.0055, 0.0055, 0.0055, 0.0055, 0.0055]
    system.properties.set_mulliken_spin_populations(spin_populations, Engine("dummy"))

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_nrad      = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_hydrogens = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_benzene_radical_anion_with_spin():

    xyz_file = f"{XYZPATH}/benzene.xyz"
    system = System.from_xyz(xyz_file, charge=-1, spin=2)

    spin_populations = [0.1667, 0.1666, 0.1667, 0.1666, 0.1667, 0.1667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    system.properties.set_mulliken_spin_populations(spin_populations, Engine("dummy"))

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_nrad      = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_hydrogens = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_carbene_triplet_with_spin():

    xyz_file = f"{XYZPATH}/carbene.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=3)

    spin_populations = [1.956, 0.022, 0.022]
    system.properties.set_mulliken_spin_populations(spin_populations, Engine("dummy"))

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0]
    expected_nrad      = [2, 0, 0]
    expected_hydrogens = [2, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_methyl_radical_with_spin():

    xyz_file = f"{XYZPATH}/methyl.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=2)

    spin_populations = [1.0, 0.0, 0.0, 0.0]
    system.properties.set_mulliken_spin_populations(spin_populations, Engine("dummy"))

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0, 0]
    expected_nrad      = [1, 0, 0, 0]
    expected_hydrogens = [3, 0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"


def test_system_to_mol_nitric_oxide_with_spin():

    xyz_file = f"{XYZPATH}/nitric_oxide.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=2)

    spin_populations = [0.4584, 0.2708, 0.2708]
    system.properties.set_mulliken_spin_populations(spin_populations, Engine("dummy"))

    try:
        mol = system_to_mol(system)
    
    except Exception as e:
        assert False, f"Exception raised on `System` to `Mol` conversion: {e}"
    
    assert type(mol) == rdchem.Mol, "ERROR: Output type is not `rdchem.Mol`"

    expected_charge    = [0, 0, 0]
    expected_nrad      = [1, 0, 0]
    expected_hydrogens = [0, 0, 0]

    atom: rdchem.Atom = None
    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetSymbol() == system.geometry.atoms[i], f"Wrong atom symbol on syte {i}"
        assert atom.GetFormalCharge() == expected_charge[i], f"Wrong formal charge on atom {i}"
        assert atom.GetNumRadicalElectrons() == expected_nrad[i], f"Wrong number of radical electrons on atom {i}"
        assert atom.GetTotalNumHs(includeNeighbors=True) == expected_hydrogens[i], f"Wrong number of hydrogens on atom {i}"
    
                                                                                                                        
