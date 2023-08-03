from spycci.systems import Ensemble
from spycci.systems import System
from spycci.core.base import Engine
from spycci.engines.xtb import XtbInput
from spycci.wrappers.crest import deprotonate
from spycci.tools.reorderenergies import reorder_energies

from spycci.constants import Eh_to_kcalmol, proton_hydration_free_energy

import logging

logger = logging.getLogger(__name__)


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

    Raises
    ------
    TypeError
        Exception raised if either protonated or deprotonated molecules are Ensembles
    RuntimeError
        Exception raised if the difference in number of atoms between protonated and
        deprotonated species is different from 1 or if a mismatch is detected between electronic
        levels of theory.

    Returns
    -------
    pKa : float
        pKa of the molecule.
    """

    if type(protonated) == Ensemble or type(deprotonated) == Ensemble:
        logger.error(
            "Calculating pKa for Ensemble instead of System. Currently not supported."
        )
        raise TypeError(
            "Calculating pKa for Ensemble instead of System. Currently not supported."
        )

    if protonated.geometry.atomcount - deprotonated.geometry.atomcount != 1:
        logger.error(f"{protonated.name} deprotomer differs for more than 1 atom.")
        raise RuntimeError(f"{protonated.name} deprotomer differs for more than 1 atom.")

    if deprotonated.charge - protonated.charge != -1:
        logger.error(
            f"{protonated.name} deprotomer differs for more than 1 unit of charge."
        )
        raise RuntimeError(
            f"{protonated.name} deprotomer differs for more than 1 unit of charge."
        )

    if (
        protonated.properties.electronic_energy is None
        or deprotonated.properties.electronic_energy is None
    ):
        raise RuntimeError("Electronic energies not found. Cannot calculate pKa.")

    if (
        protonated.properties.level_of_theory_electronic
        != deprotonated.properties.level_of_theory_electronic
    ):
        raise RuntimeError(
            "Mismatch found between electronic levels of theory. Cannot calculate pKa."
        )

    protonated_energy = protonated.properties.electronic_energy * Eh_to_kcalmol
    deprotonated_energy = deprotonated.properties.electronic_energy * Eh_to_kcalmol

    if (
        protonated.properties.level_of_theory_vibronic is not None
        and protonated.properties.level_of_theory_vibronic
        == deprotonated.properties.level_of_theory_vibronic
    ):
        protonated_energy += protonated.properties.vibronic_energy * Eh_to_kcalmol
        deprotonated_energy += deprotonated.properties.vibronic_energy * Eh_to_kcalmol

    proton_self_energy = 0

    if "gfn2" in protonated.properties.level_of_theory_electronic:
        proton_self_energy = 164.22  # kcal/mol

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

    deprotomers = deprotonate(protonated, ncores=ncores, maxcore=maxcore, solvent="water")
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
