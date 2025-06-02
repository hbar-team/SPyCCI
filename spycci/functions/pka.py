from spycci.systems import Ensemble
from spycci.systems import System
from spycci.core.base import Engine
from spycci.engines.xtb import XtbInput
from spycci.wrappers.crest import deprotonate
from spycci.tools.reorderenergies import reorder_energies

from spycci.constants import Eh_to_kcalmol, proton_hydration_free_energy

import logging

logger = logging.getLogger(__name__)


def _check_structure_acid_base_pair(protonated: System, deprotonated: System) -> None:
    """
    Checks if the provided `protonated` and `deprotonated` objects are compatible with each other and
    with a pKa calculation. If validation fails an exception is raised.

    Parameters
    ----------
    protonated : System object
        The molecule in the protonated form
    deprotonated : System object
        The molecule in the deprotonated form

    Raises
    ------
    TypeError
        Exception raised if either of the objects type is different from `System`
    RuntimeError
        Excpetion raised if the atom count and charge variation are not compatible.
    """
    # Check if the objects are instances of `System`
    if type(protonated) == Ensemble or type(deprotonated) == Ensemble:
        msg = "The calculation of pKa for Ensemble objects is currently not supported."
        logger.error(msg)
        raise TypeError(msg)
    elif type(protonated) != System or type(deprotonated) != System:
        msg = "The calculation of pKa requires System type arguments."
        logger.error(msg)
        raise TypeError(msg)

    # Check that the atomcount of the two molecules matches
    if protonated.geometry.atomcount - deprotonated.geometry.atomcount != 1:
        msg = f"{protonated.name} deprotomer differs for more than 1 atom."
        logger.error(msg)
        raise RuntimeError(msg)

    # Check that the molecular charge of the deprotomer is 1 unit less than the protonated one
    if deprotonated.charge - protonated.charge != -1:
        msg = f"{protonated.name} deprotomer differs for more than 1 unit of charge."
        logger.error(msg)
        raise RuntimeError(msg)


def _validate_acid_base_pair(protonated: System, deprotonated: System) -> bool:
    """
    Checks if the provided `protonated` and `deprotonated` objects are compatible with each other and
    with a pKa calculation and verify the matching between levels of theory used in the computation.
    If validation fails an exception is raised. The function returns a bool value correspondent to the
    availability of vibronic calculations.

    Parameters
    ----------
    protonated : System object
        The molecule in the protonated form
    deprotonated : System object
        The molecule in the deprotonated form

    Raises
    ------
    TypeError
        Exception raised if either of the objects type is different from `System`
    RuntimeError
        Excpetion raised if the electronic level of theory is not found

    Returns
    -------
    bool
        True if the two system have appropriate and matching vibronic levels of theory. False otherwise.
    """
    # Check the provided structures for type, atomcount and charge
    _check_structure_acid_base_pair(protonated, deprotonated)

    # Check that both the speces have an electronic energy value associated
    if protonated.properties.electronic_energy is None:
        msg = "Electronic energy not found for protonated molecule."
        logger.error(msg)
        raise RuntimeError(msg)

    if deprotonated.properties.electronic_energy is None:
        msg = "Electronic energy not found for deprotonated molecule."
        logger.error(msg)
        raise RuntimeError(msg)

    # Check if the electronic level of theory for both molecules match
    elot_protonated = protonated.properties.level_of_theory_electronic
    elot_deprotonated = deprotonated.properties.level_of_theory_electronic
    if elot_protonated != elot_deprotonated:
        print(elot_protonated)
        print(elot_deprotonated)
        msg = "Mismatch found between electronic levels of theory."
        logger.error(msg)
        raise RuntimeError(msg)

    # Check if both speces have matching vibronic energy values associated
    if (
        protonated.properties.vibronic_energy is not None
        and protonated.properties.level_of_theory_vibronic
        == deprotonated.properties.level_of_theory_vibronic
    ):

        if elot_protonated != protonated.properties.level_of_theory_vibronic:
            logger.warning(
                f"Vibronic level of theory ({protonated.properties.level_of_theory_vibronic}) is different form the electronic one ({elot_protonated}). The pKa calculation will be executed anyway."
            )

        return True

    else:
        logger.warning(
            "Vibronic energies not found. The pKa calculation will be executed with only electronic energies"
        )

        return False


def calculate_pka(protonated: System, deprotonated: System):
    """
    Calculates the pKa of a molecule, given the protonated and deprotonated forms assuming
    that energies have already been computed.

    Parameters
    ----------
    protonated : System object
        molecule in the protonated form
    deprotonated : System object
        molecule in the deprotonated form

    Returns
    -------
    pKa : float
        pKa of the molecule.
    """

    # Validate systems and check if vibronic energies are available
    with_vibronic = _validate_acid_base_pair(protonated, deprotonated)

    # Extract electronic energies
    protonated_energy = protonated.properties.electronic_energy * Eh_to_kcalmol
    deprotonated_energy = deprotonated.properties.electronic_energy * Eh_to_kcalmol

    # If available consider vibronic energies
    if with_vibronic:
        protonated_energy += protonated.properties.vibronic_energy * Eh_to_kcalmol
        deprotonated_energy += deprotonated.properties.vibronic_energy * Eh_to_kcalmol

    # If gfn2 from xTB is used consider an additional correction factor for the proton self energy
    proton_self_energy = 0
    if "gfn2" in protonated.properties.level_of_theory_electronic:
        proton_self_energy = 164.22  # kcal/mol

    # Compute the pKa and set it as a property of the protonated molecule
    pka = (
        (
            deprotonated_energy
            + (proton_hydration_free_energy + proton_self_energy)
            - protonated_energy
        )
    ) / (2.303 * 1.98720425864083 / 1000 * 298.15)

    protonated.properties.set_pka(
        value=pka,
        electronic_engine=protonated.properties.level_of_theory_electronic,
        vibronic_engine=protonated.properties.level_of_theory_vibronic,
    )

    return pka


def auto_calculate_pka(
    protonated: System,
    method_el: Engine,
    method_vib: Engine = None,
    method_opt: Engine = None,
    ncores: int = None,
    maxcore: int = 350,
):
    """
    Automatically calculates the pKa of a given `protonated` molecule. The routine computes
    all the deprotomers of the molecule using CREST, orders the deprotomers according to
    their energy computed with a user-defined level of theory and calculates the pKa.

    Arguments
    ---------
    protonated: System
        The protonated molecule for which the pKa must be computed
    method_el: Engine
        The computational engine to be used in the electronic level of theory calculations
    method_vib: Engine (optional)
        The computational engine to be used in the vibronic level of theory calculations. If
        set to None (default) the pKa will be computed without the vibronic contributions.
    method_opt: Engine (optional)
        The computational engine to be used in the geometry optimization of the protonated
        molecule and its deprotomers. If set to None (default) will use xTB gfn2 and the
        alpb solvent model for water.
    ncores: int (optional)
        The number of cores to be used in the calculations. If set to None (default) will use
        the maximun number of available cores.
    maxcore: int (optional)
        For the engines that supprots it, the memory assigned to each core used in the
        computation.

    Returns
    -------
    float
        pKa of the molecule.
    System
        the structure of the considered deprotomer.
    """

    if method_opt is None:
        method_opt = XtbInput(solvent="water")

    method_opt.opt(protonated, inplace=True, ncores=ncores, maxcore=maxcore)

    if method_vib == method_el:
        method_el.freq(protonated, inplace=True, ncores=ncores, maxcore=maxcore)
    else:
        method_el.spe(protonated, inplace=True, ncores=ncores, maxcore=maxcore)

        if method_vib is not None and method_vib != method_opt:
            dummy = method_vib.freq(protonated, ncores=ncores, maxcore=maxcore)
            protonated.properties.set_vibronic_energy(
                dummy.properties.vibronic_energy, method_vib
            )

    deprotomers = deprotonate(
        protonated, ncores=ncores, maxcore=maxcore, solvent="water"
    )
    ordered_deprotomers = reorder_energies(
        deprotomers.systems,
        ncores=ncores,
        maxcore=maxcore,
        method_opt=method_opt,
        method_el=method_el,
        method_vib=method_vib,
    )

    lowest_deprotomer = ordered_deprotomers[0]

    pka = calculate_pka(protonated, lowest_deprotomer)

    return pka, lowest_deprotomer
