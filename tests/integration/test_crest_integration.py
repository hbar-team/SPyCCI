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

    mol = System(f"{TEST_DIR}/utils/xyz_files/urea.xyz")

    try:
        tautomers: Ensemble = crest.tautomer_search(mol, ncores=4, solvent="water", optionals="--mquick")
    except:
        assert False, "Unexpected exception raised during tautomer search"

    else:
        assert len(tautomers.systems) == 5

        rmtree("output_files")
        rmtree("error_files")


# Test the conformer_search() function on a propanol molecule in water
def test_crest_conformer_search():

    mol = System(f"{TEST_DIR}/utils/xyz_files/propan-1-ol.xyz")

    try:
        conformers: Ensemble = crest.conformer_search(mol, ncores=4, solvent="water", optionals="--mquick")
    except:
        assert False, "Unexpected exception raised during tautomer search"

    else:
        assert len(conformers.systems) in [5, 6, 7]

        rmtree("output_files")
        rmtree("error_files")


# Test the deprotonate() function on a tyrosine molecule in water
def test_crest_deprotonate():

    mol = System(f"{TEST_DIR}/utils/xyz_files/3-amino-L-tyrosine.xyz")

    try:
        conformers: Ensemble = crest.deprotonate(mol, ncores=4, solvent="water", optionals="--mquick")
    except:
        assert False, "Unexpected exception raised during tautomer search"

    else:
        assert len(conformers.systems) == 2

        rmtree("output_files")
        rmtree("error_files")


# Test the protonate() function on a tyrosine molecule in water
def test_crest_protonate():

    mol = System(f"{TEST_DIR}/utils/xyz_files/3-amino-L-tyrosine.xyz")

    try:
        conformers: Ensemble = crest.protonate(mol, ncores=4, solvent="water", optionals="--mquick")
    except:
        assert False, "Unexpected exception raised during tautomer search"

    else:
        assert len(conformers.systems) == 9

        rmtree("output_files")
        rmtree("error_files")


##################################################################################################
# NOTE: This test segfaults with low-memory machines, even though it shouldn't need as much RAM. #
# Works without needing to change stack sizes if there's enough system memory, still it's safer  #
# to skip it for now                                                                             #
##################################################################################################
# Test the qcg_grow() function on a urea molecule + 5 water molecules
@pytest.mark.skip(reason="This test is currently failing with a segfault")
def test_qcg_grow():

    solute = System(f"{TEST_DIR}/utils/xyz_files/urea.xyz")
    solvent = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        cluster: System = crest.qcg_grow(
            solute=solute,
            solvent=solvent,
            ncores=4,
            alpb_solvent="water",
            nsolv=5,
            optionals="--mquick",
        )
    except:
        assert False, "Unexpected exception raised during QCG run"

    else:
        assert cluster.geometry.atomcount == 23

        rmtree("output_files")
        rmtree("error_files")


###############################################################################################
# NOTE: This test is still crashing with a segfault, even with an OMP_STACKSIZE of 10GB       #
# on a system with a 64-core Epyc CPU and 512GB of memory, so this looks like a CREST problem #
###############################################################################################


# Test the qcg_ensemble() function on a urea molecule + 3 water molecules
@pytest.mark.skip(reason="This test is currently failing with a segfault")
def test_qcg_ensemble():

    solute = System(f"{TEST_DIR}/utils/xyz_files/urea.xyz")
    solvent = System(f"{TEST_DIR}/utils/xyz_files/water.xyz")

    try:
        ensemble: Ensemble = crest.qcg_ensemble(
            solute=solute,
            solvent=solvent,
            ncores=4,
            alpb_solvent="water",
            nsolv=5,
            optionals="--mquick",
        )
    except:
        assert False, "Unexpected exception raised during tautomer search"

    else:
        assert len(ensemble.systems) == 10

        rmtree("output_files")
        rmtree("error_files")
