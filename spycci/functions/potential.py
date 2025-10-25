from spycci.systems import Ensemble
from spycci.systems import System
from spycci.constants import Eh_to_kcalmol
from typing import Tuple

import logging

logger = logging.getLogger(__name__)


def calculate_reduction_potential(
    oxidised: System,
    reduced: System,
    pH: float = 7.0,
) -> Tuple[float, int, int]:
    """
    Calculates the reduction potential of a molecule, given the oxidized and reduced forms
    assuming that energies have already been computed. The number of electrons exchanged in
    the process is automatically computed.

    Parameters
    ----------
    oxidised : System object
        molecule in the oxidised state
    reduced : System object
        molecule in the reduced state
    pH : float
        pH at which the redox potential is evaluated, by default pH=7. This is important in
        case of a proton-coupled electron transfer (PCET) redox process

    Raises
    ------
    TypeError
        Exception raised if either oxidised or reduced molecules are Ensembles
    RuntimeError
        Exception raised if a mismatch is detected between electronic levels of theory.

    Returns
    -------
    float
        reduction potential of the molecule.
    int
        number of exchanged electrons
    int
        number of exchanged protons
    """

    if type(oxidised) == Ensemble or type(oxidised) == Ensemble:
        logger.error(
            "Calculating pKa for Ensemble instead of System. Currently not supported."
        )
        raise TypeError(
            "Calculating pKa for Ensemble instead of System. Currently not supported."
        )

    oxidised_protons = oxidised.geometry.atoms.count("H")
    reduced_protons = reduced.geometry.atoms.count("H")
    exchanged_protons = reduced_protons - oxidised_protons

    exchanged_electrons = (oxidised.charge - reduced.charge) + exchanged_protons
    if exchanged_protons != 0:
        logger.info(
            f"pH = {pH}: {exchanged_electrons}e/{exchanged_protons}H PCET reduction detected. Calculated potential is pH-dependent."
        )
    else:
        logger.info(f"pH = {pH}: {exchanged_electrons}e reduction detected.")

    if reduced.geometry.atomcount - oxidised.geometry.atomcount != exchanged_protons:
        logger.error(
            "oxidised and reduced forms are not the same molecule. Only molecules differing for the number of protons are allowed."
        )
        raise RuntimeError(
            "oxidised and reduced forms are not the same molecule. Only molecules differing for the number of protons are allowed."
        )

    if exchanged_electrons < 0:
        logger.error(
            "Number of exchanged electrons is inconsistent with a reduction reaction. Make sure you entered the oxidised and reduced forms in the correct order."
        )
        raise RuntimeError(
            "Number of exchanged electrons is inconsistent with a reduction reaction. Make sure you entered the oxidised and reduced forms in the correct order."
        )
    elif exchanged_electrons == 0:
        logger.error(
            "No difference in number of electrons has been detected between oxidised and reduced species."
        )
        raise RuntimeError(
            "No difference in number of electrons has been detected between oxidised and reduced species."
        )

    if (
        oxidised.properties.electronic_energy is None
        or reduced.properties.electronic_energy is None
    ):
        raise RuntimeError(
            "Electronic energies not found. Cannot calculate reduction potential."
        )

    if (
        oxidised.properties.level_of_theory_electronic
        != reduced.properties.level_of_theory_electronic
    ):
        raise RuntimeError(
            "Mismatch found between electronic levels of theory. Cannot calculate reduction potential."
        )

    oxidised_energy = oxidised.properties.electronic_energy * Eh_to_kcalmol
    reduced_energy = reduced.properties.electronic_energy * Eh_to_kcalmol

    if (
        oxidised.properties.level_of_theory_vibrational is not None
        and oxidised.properties.level_of_theory_vibrational
        == reduced.properties.level_of_theory_vibrational
    ):
        oxidised_energy += oxidised.properties.free_energy_correction * Eh_to_kcalmol
        reduced_energy += reduced.properties.free_energy_correction * Eh_to_kcalmol

    reference_potential = 4.28  # V, SHE reference

    proton_self_energy = 0
    electron_self_energy = 0

    if "gfn2" in oxidised.properties.level_of_theory_electronic:
        electron_self_energy = 111.75  # kcal/mol
        proton_self_energy = 164.22  # kcal/mol

    potential = (
        -(4184 / (96485 * exchanged_electrons))
        * (
            reduced_energy
            + electron_self_energy * exchanged_electrons
            - (oxidised_energy + (-270.29 + proton_self_energy) * exchanged_protons)
        )
        - reference_potential
        - 0.059 * pH * exchanged_protons
    )

    return potential, exchanged_electrons, exchanged_protons
