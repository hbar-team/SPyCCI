import pytest, tempfile, os

from os.path import abspath, dirname, join
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from spycci.systems import System
from spycci.core.geometry import MolecularGeometry
from spycci.core.cheminformatics import ChemInfo

from spycci.core.base import Engine
from rdkit.Chem import rdchem

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))
XYZPATH = join(TEST_DIR, "utils/xyz_examples")

#####################################################################################################
#                    TESTS RELATED TO RDKIT INTERFACE AND CONNECTIVITY ROUTINES                     #
#####################################################################################################

def test_System_connectivity_simple():

    mol = System.from_smiles("formaldehyde", "C=O")

    try:
        wrapper = ChemInfo(mol)
        adj, bt = wrapper.get_connectivity()
    
    except Exception as e:
        assert False, f"Exception raised on `get_connectivity` call: {e}"

    expected_adj = [
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]

    expected_bt = [
        [0.0, 2.0, 1.0, 1.0],
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]

    assert_array_almost_equal(adj, expected_adj, decimal=6)
    assert_array_almost_equal(bt, expected_bt, decimal=6)


def test_System_connectivity_charged():

    geom = MolecularGeometry()
    geom.append("C", [-0.5, 0.0, 0.0])
    geom.append("N", [0.5, 0.0, 0.0])

    mol = System("cyanide", geom, charge=-1)

    try:
        wrapper = ChemInfo(mol)
        adj, bt = wrapper.get_connectivity()
    
    except Exception as e:
        assert False, f"Exception raised on `get_connectivity` call: {e}"

    assert_array_almost_equal(adj, [[0.0, 1.0], [1.0, 0.0]], decimal=6)
    assert_array_almost_equal(bt, [[0.0, 3.0], [3.0, 0.0]], decimal=6)


def test_System_connectivity_localized_radical():

    mol = System.from_smiles("methyl radical", "[CH3]", spin=2)

    try:
        wrapper = ChemInfo(mol)
        adj, bt = wrapper.get_connectivity()
    
    except Exception as e:
        assert False, f"Exception raised on `get_connectivity` call: {e}"

    expected_adj = [
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]

    expected_bt = [
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]

    assert_array_almost_equal(adj, expected_adj, decimal=6)
    assert_array_almost_equal(bt, expected_bt, decimal=6)


def test_System_connectivity_delocalized_radical():

    mol = System.from_smiles("benzyl radical", "[CH2]c1ccccc1", spin=2)

    try:
        wrapper = ChemInfo(mol)
        adj, bt = wrapper.get_connectivity()
    
    except Exception as e:
        assert False, f"Exception raised on `get_connectivity` call: {e}"

    expected_adj = [
        [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,],
        [1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,],
        [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,],
        [0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,],
        [0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0.,],
        [0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,],
    ]

    expected_bt = [
        [0., 2., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,],
        [2., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 1., 0., 2., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,],
        [0., 0., 2., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,],
        [0., 0., 0., 1., 0., 2., 0., 0., 0., 0., 0., 1., 0., 0.,],
        [0., 0., 0., 0., 2., 0., 1., 0., 0., 0., 0., 0., 1., 0.,],
        [0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,],
    ]

    assert_array_almost_equal(adj, expected_adj, decimal=6)
    assert_array_almost_equal(bt, expected_bt, decimal=6)


def test_System_connectivity_delocalized_cation():

    mol = System.from_smiles("benzyl cation", "[CH2]c1ccccc1", charge=1, spin=1)

    try:
        wrapper = ChemInfo(mol)
        adj, bt = wrapper.get_connectivity()
    
    except Exception as e:
        assert False, f"Exception raised on `get_connectivity` call: {e}"

    expected_adj = [
        [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,],
        [1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,],
        [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,],
        [0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,],
        [0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0.,],
        [0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,],
    ]

    expected_bt = [
        [0., 2., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,],
        [2., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 1., 0., 2., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,],
        [0., 0., 2., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,],
        [0., 0., 0., 1., 0., 2., 0., 0., 0., 0., 0., 1., 0., 0.,],
        [0., 0., 0., 0., 2., 0., 1., 0., 0., 0., 0., 0., 1., 0.,],
        [0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,],
    ]

    assert_array_almost_equal(adj, expected_adj, decimal=6)
    assert_array_almost_equal(bt, expected_bt, decimal=6)


def test_System_connectivity_carbene():

    mol = System.from_smiles("methylene", "[CH2]")

    try:
        wrapper = ChemInfo(mol)
        adj, bt = wrapper.get_connectivity()
    
    except Exception as e:
        assert False, f"Exception raised on `get_connectivity` call: {e}"

    expected_adj = [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    expected_bt = [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    assert_array_almost_equal(adj, expected_adj, decimal=6)
    assert_array_almost_equal(bt, expected_bt, decimal=6)


def test_System_connectivity_fragments():

    geom = MolecularGeometry()
    geom.append("C", [-4.21089,  1.17290, 0.00000])
    geom.append("O", [-3.80998,  0.01973, 0.00000])
    geom.append("O", [-3.38675,  2.23147, 0.00000])
    geom.append("H", [-2.45844,  1.87823, 0.00000])
    geom.append("O", [-1.07460,  1.00027, 0.00000])
    geom.append("C", [-0.67368, -0.15290, 0.00000])
    geom.append("O", [-1.49782, -1.21147, 0.00000])
    geom.append("H", [ 0.37934, -0.47119, 0.00000])
    geom.append("H", [-2.42614, -0.85823, 0.00000])
    geom.append("H", [-5.26392,  1.49119, 0.00000])

    mol = System("dimer", geom)

    try:
        wrapper = ChemInfo(mol)
        adj, bt = wrapper.get_connectivity()
    
    except Exception as e:
        assert False, f"Exception raised on `get_connectivity` call: {e}"

    expected_adj = [
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    expected_bt = [
        [0., 2., 1., 0., 0., 0., 0., 0., 0., 1.,],
        [2., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 0., 2., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 2., 0., 1., 1., 0., 0.,],
        [0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
    ]

    assert_array_almost_equal(adj, expected_adj, decimal=6)
    assert_array_almost_equal(bt, expected_bt, decimal=6)


#####################################################################################################
#                               TESTS RELATED TO `save_sdf` FUNCTION                                #
#####################################################################################################


def test_save_sdf_simple_molecule():
    
    system = System.from_smiles("methane", "C")

    expected = []
    expected.append("")
    expected.append("     RDKit          3D")
    expected.append("")
    expected.append("  5  4  0  0  0  0  0  0  0  0999 V2000")
    expected.append("    0.0000    0.0000   -0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -0.5870    0.8972    0.2082 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -0.6306   -0.7405   -0.4968 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("    0.3776   -0.4137    0.9376 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("    0.8400    0.2570   -0.6490 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("  2  1  1  0")
    expected.append("  3  1  1  0")
    expected.append("  4  1  1  0")
    expected.append("  5  1  1  0")
    expected.append("M  END")
    expected.append("$$$$")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as tmp:
        sdf_path = tmp.name

    try:
        wrapper = ChemInfo(system)
        wrapper.save_sdf(sdf_path)

        with open(sdf_path, "r") as file:
            for i, line in enumerate(file):
                assert line.strip('\n') == expected[i]

    finally:
        os.remove(sdf_path)


def test_save_sdf_carbene_singlet():
    
    xyz_file = f"{XYZPATH}/carbene.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=1)

    expected = []
    expected.append("")
    expected.append("     RDKit          3D")
    expected.append("")
    expected.append("  3  2  0  0  0  0  0  0  0  0999 V2000")
    expected.append("   -5.1911   -0.7092   -0.2368 C   0  0  0  0  0  2  0  0  0  0  0  0")
    expected.append("   -4.8221   -0.5261    0.7745 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -4.8221    0.0869   -0.8869 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("  2  1  1  0")
    expected.append("  3  1  1  0")
    expected.append("M  END")
    expected.append("$$$$")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as tmp:
        sdf_path = tmp.name

    try:
        wrapper = ChemInfo(system)
        wrapper.save_sdf(sdf_path)

        with open(sdf_path, "r") as file:
            for i, line in enumerate(file):
                assert line.strip('\n') == expected[i]

    finally:
        os.remove(sdf_path)


def test_save_sdf_carbene_triplet():
    
    xyz_file = f"{XYZPATH}/carbene.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=3)

    expected = []
    expected.append("")
    expected.append("     RDKit          3D")
    expected.append("")
    expected.append("  3  2  0  0  0  0  0  0  0  0999 V2000")
    expected.append("   -5.1911   -0.7092   -0.2368 C   0  0  0  0  0  2  0  0  0  0  0  0")
    expected.append("   -4.8221   -0.5261    0.7745 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -4.8221    0.0869   -0.8869 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("  2  1  1  0")
    expected.append("  3  1  1  0")
    expected.append("M  RAD  1   1   3")
    expected.append("M  END")
    expected.append("$$$$")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as tmp:
        sdf_path = tmp.name

    try:
        wrapper = ChemInfo(system)
        wrapper.save_sdf(sdf_path)

        with open(sdf_path, "r") as file:
            for i, line in enumerate(file):
                assert line.strip('\n') == expected[i]

    finally:
        os.remove(sdf_path)


def test_save_sdf_methyl_radical():
    
    xyz_file = f"{XYZPATH}/methyl.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=2)

    expected = []
    expected.append("")
    expected.append("     RDKit          3D")
    expected.append("")
    expected.append("  4  3  0  0  0  0  0  0  0  0999 V2000")
    expected.append("   -4.6201    0.4279    0.9279 C   0  0  0  0  0  3  0  0  0  0  0  0")
    expected.append("   -3.5995    0.4728    1.2859 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -5.1305   -0.5247    0.8653 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -5.1305    1.3358    0.6324 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("  2  1  1  0")
    expected.append("  3  1  1  0")
    expected.append("  4  1  1  0")
    expected.append("M  RAD  1   1   2")
    expected.append("M  END")
    expected.append("$$$$")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as tmp:
        sdf_path = tmp.name

    try:
        wrapper = ChemInfo(system)
        wrapper.save_sdf(sdf_path)

        with open(sdf_path, "r") as file:
            for i, line in enumerate(file):
                assert line.strip('\n') == expected[i]

    finally:
        os.remove(sdf_path)


def test_save_sdf_methyl_cation():
    
    xyz_file = f"{XYZPATH}/methyl.xyz"
    system = System.from_xyz(xyz_file, charge=1, spin=1)

    expected = []
    expected.append("")
    expected.append("     RDKit          3D")
    expected.append("")
    expected.append("  4  3  0  0  0  0  0  0  0  0999 V2000")
    expected.append("   -4.6201    0.4279    0.9279 C   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -3.5995    0.4728    1.2859 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -5.1305   -0.5247    0.8653 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -5.1305    1.3358    0.6324 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("  2  1  1  0")
    expected.append("  3  1  1  0")
    expected.append("  4  1  1  0")
    expected.append("M  CHG  1   1   1")
    expected.append("M  END")
    expected.append("$$$$")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as tmp:
        sdf_path = tmp.name

    try:
        wrapper = ChemInfo(system)
        wrapper.save_sdf(sdf_path)

        with open(sdf_path, "r") as file:
            for i, line in enumerate(file):
                assert line.strip('\n') == expected[i]

    finally:
        os.remove(sdf_path)


def test_save_sdf_methyl_anion():
    
    xyz_file = f"{XYZPATH}/methyl.xyz"
    system = System.from_xyz(xyz_file, charge=-1, spin=1)

    expected = []
    expected.append("")
    expected.append("     RDKit          3D")
    expected.append("")
    expected.append("  4  3  0  0  0  0  0  0  0  0999 V2000")
    expected.append("   -4.6201    0.4279    0.9279 C   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -3.5995    0.4728    1.2859 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -5.1305   -0.5247    0.8653 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -5.1305    1.3358    0.6324 H   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("  2  1  1  0")
    expected.append("  3  1  1  0")
    expected.append("  4  1  1  0")
    expected.append("M  CHG  1   1  -1")
    expected.append("M  END")
    expected.append("$$$$")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as tmp:
        sdf_path = tmp.name

    try:
        wrapper = ChemInfo(system)
        wrapper.save_sdf(sdf_path)

        with open(sdf_path, "r") as file:
            for i, line in enumerate(file):
                assert line.strip('\n') == expected[i]

    finally:
        os.remove(sdf_path)


def test_save_sdf_nitric_oxide():
    
    xyz_file = f"{XYZPATH}/nitric_oxide.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=2)

    spin_populations = [0.4584, 0.2708, 0.2708]
    system.properties.set_mulliken_spin_populations(spin_populations, Engine("dummy"))

    expected = []
    expected.append("")
    expected.append("     RDKit          3D")
    expected.append("")
    expected.append("  3  2  0  0  0  0  0  0  0  0999 V2000")
    expected.append("   -0.1221    1.1355   -0.9373 N   0  0  0  0  0  4  0  0  0  0  0  0")
    expected.append("    0.3341   -0.1096   -1.0998 O   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("   -0.5288    1.3363    0.3192 O   0  0  0  0  0  0  0  0  0  0  0  0")
    expected.append("  2  1  2  0")
    expected.append("  3  1  2  0")
    expected.append("M  RAD  1   1   2")
    expected.append("M  END")
    expected.append("$$$$")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as tmp:
        sdf_path = tmp.name

    try:
        wrapper = ChemInfo(system)
        wrapper.save_sdf(sdf_path)

        with open(sdf_path, "r") as file:
            for i, line in enumerate(file):
                assert line.strip('\n') == expected[i]

    finally:
        os.remove(sdf_path)


#####################################################################################################
#                        TESTS RELATED TO `locate_hydrogen_bonds` FUNCTION                          #
#####################################################################################################

def test_locate_hydrogen_bonds():
    
    xyz_file = f"{XYZPATH}/malondialdehyde.xyz"
    system = System.from_xyz(xyz_file, charge=0, spin=1)

    try:
        wrapper = ChemInfo(system)
        hbonds = wrapper.locate_hydrogen_bonds()
    
    except Exception as e:
        assert False, f"Exception raised on `locate_hydrogen_bonds` call: {e}"
    
    assert hbonds == [[4, 5]], "Wrong hydrogen bond detected"