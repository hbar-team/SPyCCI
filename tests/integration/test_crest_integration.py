import pytest

from spycci.wrappers import crest
from spycci.systems import System, Ensemble
from os.path import dirname, abspath
from shutil import rmtree

# =================================================================
#     The following test has been developed for CREST 3.0.2
# =================================================================

# Get the path of the tests directory
TEST_DIR = dirname(abspath(__file__))

# Test the tautomer_search() function on a urea molecule in water
def test_crest_tautomer_search():

    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/urea.xyz")

    try:
        tautomers: Ensemble = crest.tautomer_search(mol, ncores=4, solvent="water")
    except:
        assert False, "Unexpected exception raised during tautomer search"

    else:
        assert len(tautomers.systems) == 5

        rmtree("output_files")
        rmtree("error_files")


# Test the conformer_search() function on a propanol molecule in water
def test_crest_conformer_search():

    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/propan-1-ol.xyz")

    try:
        conformers: Ensemble = crest.conformer_search(mol, ncores=4, solvent="water")
    except:
        assert False, "Unexpected exception raised during tautomer search"

    else:
        assert len(conformers.systems) in [5, 6, 7]

        rmtree("output_files")
        rmtree("error_files")


# Test the deprotonate() function on a tyrosine molecule in water
def test_crest_deprotonate():

    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/3-amino-L-tyrosine.xyz")

    try:
        conformers: Ensemble = crest.deprotonate(mol, ncores=4, solvent="water")
    except:
        assert False, "Unexpected exception raised during tautomer search"

    else:
        assert len(conformers.systems) == 2

        rmtree("output_files")
        rmtree("error_files")


# Test the protonate() function on a tyrosine molecule in water
def test_crest_protonate():

    mol = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/3-amino-L-tyrosine.xyz")

    try:
        conformers: Ensemble = crest.protonate(mol, ncores=4, solvent="water")
    except:
        assert False, "Unexpected exception raised during tautomer search"

    else:
        assert len(conformers.systems) == 9

        rmtree("output_files")
        rmtree("error_files")


### !!! ###
# QCG is currently crashing on my workstation, do not use these test until fixed!
### !!! ###

# # Test the qcg_grow() function on a urea molecule + 5 water molecules
# def test_qcg_grow():

#     solute = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/urea.xyz")
#     solvent = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

#     try:
#         cluster: System = crest.qcg_grow(
#             solute=solute, solvent=solvent, ncores=4, alpb_solvent="water"
#         )
#     except:
#         assert False, "Unexpected exception raised during QCG run"

#     else:
#         assert cluster.geometry.atomcount == 23

#         rmtree("output_files")
#         rmtree("error_files")


# # Test the qcg_ensemble() function on a urea molecule + 3 water molecules
# def test_qcg_ensemble():

#     solute = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/urea.xyz")
#     solvent = System.from_xyz(f"{TEST_DIR}/utils/xyz_files/water.xyz")

#     try:
#         ensemble: Ensemble = crest.qcg_ensemble(
#             solute=solute, solvent=solvent, ncores=4, alpb_solvent="water"
#         )
#     except:
#         assert False, "Unexpected exception raised during tautomer search"

#     else:
#         assert len(ensemble.systems) == ??

#         rmtree("output_files")
#         rmtree("error_files")
