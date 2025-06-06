import logging, sh, shutil, os
from tempfile import mkdtemp
from copy import deepcopy
from typing import Union, Optional, Dict, Tuple

from spycci.systems import Ensemble
from spycci.systems import System
from spycci.core.base import Engine
from spycci.engines.xtb import XtbInput
from spycci.engines.orca import OrcaInput
from spycci.wrappers.crest import deprotonate
from spycci.tools.reorderenergies import reorder_energies

import spycci.constants as scon
import numpy as np

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


def _validate_acid_base_pair(
    protonated: System,
    deprotonated: System,
    water: Optional[System] = None,
    oxonium: Optional[System] = None,
) -> bool:
    """
    Checks if the provided `protonated` and `deprotonated` objects are compatible with each other and
    with a pKa calculation and verify the matching between levels of theory used in the computation.
    The function also accepts a `water` and `oxonium` object as optionals to verify the compatibility
    of the levels of theory in case of the oxonium scheme. If validation fails an exception is raised.
    The function returns a bool value correspondent to the availability of vibronic calculations.

    Parameters
    ----------
    protonated : System object
        The molecule in the protonated form
    deprotonated : System object
        The molecule in the deprotonated form
    water : Optional[System]
        The water molecule to be used in the calculation
    oxonium : Optional[System]
        The oxonium molecule to be used in the calculation

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
    # Check if the user provided both water and oxonium and check the structures
    if oxonium is None and water is None:
        pass
    elif None in [oxonium, water]:
        msg = "water and oxonium molecules must be checked simultaneously."
        logger.error(msg)
        raise RuntimeError(msg)

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
        msg = "Mismatch found between electronic levels of theory."
        logger.error(msg)
        raise RuntimeError(msg)

    # Check, if povided, the water and oxonium molecules
    if oxonium and water:

        # Check the provided structures for type, atomcount and charge
        _check_structure_acid_base_pair(oxonium, water)

        # Check that both the speces have an electronic energy value associated
        if water.properties.electronic_energy is None:
            msg = "Electronic energy not found for water molecule."
            logger.error(msg)
            raise RuntimeError(msg)

        if oxonium.properties.electronic_energy is None:
            msg = "Electronic energy not found for oxonium ion."
            logger.error(msg)
            raise RuntimeError(msg)

        # Check if the electronic level of theory for both molecules match
        elot_water = water.properties.level_of_theory_electronic
        elot_oxonium = oxonium.properties.level_of_theory_electronic
        if elot_water != elot_oxonium or elot_water != elot_protonated:
            msg = "Mismatch found between electronic levels of theory."
            logger.error(msg)
            raise RuntimeError(msg)

    # Check if the speces have vibronic energy values associated
    if (
        protonated.properties.vibronic_energy is None
        or deprotonated.properties.vibronic_energy is None
    ):
        msg = "Vibronic energies not found. The pKa calculation will be executed with only electronic energies"
        logger.warning(msg)
        return False

    if oxonium and water:

        if (
            water.properties.vibronic_energy is None
            or oxonium.properties.vibronic_energy is None
        ):
            msg = "Vibronic energies not found. The pKa calculation will be executed with only electronic energies"
            logger.warning(msg)
            return False

    # Check if the level
    vlot_protonated = protonated.properties.level_of_theory_vibronic
    vlot_deprotonated = deprotonated.properties.level_of_theory_vibronic
    if vlot_protonated == vlot_deprotonated:

        if oxonium and water:
            vlot_water = water.properties.level_of_theory_vibronic
            vlot_oxonium = oxonium.properties.level_of_theory_vibronic
            if vlot_protonated != vlot_water or vlot_protonated != vlot_oxonium:
                msg = "Vibronic energies not found. The pKa calculation will be executed with only electronic energies"
                logger.warning(msg)
                return False

        if elot_protonated != vlot_protonated:
            msg = f"Vibronic level of theory ({vlot_protonated}) is different form the electronic one ({elot_protonated}). The pKa calculation will be executed anyway."
            logger.warning(msg)

        return True

    else:
        msg = "Vibronic energies not found. The pKa calculation will be executed with only electronic energies"
        logger.warning(msg)
        return False


def calculate_pka(protonated: System, deprotonated: System):
    """
    Calculates the pKa of a molecule using the direct method. The function expects as arguments
    the protonated and deprotonated forms of the molecule as a `System` objects for which the
    energies must have already been computed.

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
    protonated_energy = protonated.properties.electronic_energy * scon.Eh_to_kcalmol
    deprotonated_energy = deprotonated.properties.electronic_energy * scon.Eh_to_kcalmol

    # If available consider vibronic energies
    if with_vibronic:
        protonated_energy += protonated.properties.vibronic_energy * scon.Eh_to_kcalmol
        deprotonated_energy += (
            deprotonated.properties.vibronic_energy * scon.Eh_to_kcalmol
        )

    # If gfn2 from xTB is used consider an additional correction factor for the proton self energy
    proton_self_energy = 0
    if "gfn2" in protonated.properties.level_of_theory_electronic:
        proton_self_energy = 164.22  # kcal/mol

    # Compute the pKa and set it as a property of the protonated molecule
    pka = (
        (
            deprotonated_energy
            + (scon.proton_hydration_free_energy + proton_self_energy)
            - protonated_energy
        )
    ) / (np.log(10.0) * scon.R * 298.15 / scon.kcal_to_J)

    protonated.properties.set_pka(
        value=pka,
        electronic_engine=protonated.properties.level_of_theory_electronic,
        vibronic_engine=protonated.properties.level_of_theory_vibronic,
    )

    return pka


def calculate_pka_oxonium_scheme(
    protonated: System, deprotonated: System, water: System, oxonium: System
):
    """
    Calculates the pKa of a molecule using the oxonium method. The function expects as arguments
    the protonated and deprotonated forms of the molecule as a `System` objects for which the
    energies must have already been computed.

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
    protonated_energy = protonated.properties.electronic_energy * scon.Eh_to_kcalmol
    deprotonated_energy = deprotonated.properties.electronic_energy * scon.Eh_to_kcalmol
    water_energy = water.properties.electronic_energy * scon.Eh_to_kcalmol
    oxonium_energy = oxonium.properties.electronic_energy * scon.Eh_to_kcalmol

    # If available consider vibronic energies
    if with_vibronic:
        protonated_energy += protonated.properties.vibronic_energy * scon.Eh_to_kcalmol
        deprotonated_energy += (
            deprotonated.properties.vibronic_energy * scon.Eh_to_kcalmol
        )
        water_energy += water.properties.vibronic_energy * scon.Eh_to_kcalmol
        oxonium_energy += oxonium.properties.vibronic_energy * scon.Eh_to_kcalmol

    # Compute the pKa and set it as a property of the protonated molecule
    pka = (
        (deprotonated_energy + oxonium_energy - protonated_energy - water_energy)
    ) / (np.log(10.0) * scon.R * 298.15 / scon.kcal_to_J) - np.log10(997.0 / 18.01528)

    protonated.properties.set_pka(
        value=pka,
        electronic_engine=protonated.properties.level_of_theory_electronic,
        vibronic_engine=protonated.properties.level_of_theory_vibronic,
    )

    return pka


def run_pKa_workflow(
    protonated: System,
    deprotonated: System,
    method_geometry: Union[XtbInput, OrcaInput],
    method_energy: Union[XtbInput, OrcaInput],
    use_cosmors: bool = False,
    ncores: Optional[int] = None,
    maxcore: int = 350,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    The function runs a complete pKa workflow and retuns the pKa computed with different schemes and
    all the computed free energies. The method runs a geometry optimization of both the protonated and
    deprotonated species in solvent (water). On the obtained geometries a frequecy calculation is carried
    out to compute the Gibbs free energies for all the species. Water and Oxonium ion are automatically
    generated and optimized. The function then computes the pKa using the direct scheme and the oxonium 
    scheme. If the `use_cosmors` option is used, the function also computes frequencies in vacuum and
    solvation free energies using OpenCOSMO-RS interface (requires orca>=6.0.0). The COSMO-RS calculation
    are then used to obtain the corresponding pKa using the oxonium method.

    Arguments
    ---------
    protonated : System
        The protonated system for which the pKa must be computed
    deprotonated : System
        The deprotomer generated during the dissociation reaction
    method_geometry : Union[XtbInput, OrcaInput]
        The engine to be used to run the geometry optimizations
    method_energy : Union[XtbInput, OrcaInput]
        The engine to be used to run the energy/frequency calculations
    use_cosmors : bool
        If set to `True` will use OpenCOSMO-RS to compute solvation energies
    ncores : Optional[int]
        The number of cores to be used in the calculations. If set to `None` (default) will use
        the maximun number of available cores.
    maxcore: int (optional)
        For the engines that supprots it, the memory assigned to each core used in the
        computation.

    Retruns
    -------
    Dict[str, float]
        The dictionary containing all the computed pKa labelled by the name of the scheme employed.
        The possible labels are: 'direct', 'oxonium' and 'oxonium COSMO-RS'.
    Dict[str, float]
        The dictionary containing all the Gibbs Free energies used in the computations.
    """
    # Check if the given structures are compatible with a pKa calculation
    _check_structure_acid_base_pair(protonated, deprotonated)

    # Check if the method to be used for geometry optimization is xtb or orca
    if type(method_geometry) not in [XtbInput, OrcaInput]:
        raise TypeError("The geometry optimization method must be xTB or orca.")

    # Check if the method to be used for energy calculations is xtb or orca
    if type(method_energy) not in [XtbInput, OrcaInput]:
        raise TypeError("The geometry optimization method must be xTB or orca.")

    # Define structures of water and oxonium ion
    WATER = [
        ["O", -5.02534, 1.26595, 0.01097],
        ["H", -4.05210, 1.22164, -0.01263],
        ["H", -5.30240, 0.44124, -0.42809],
    ]

    OXONIUM = [
        ["O", -7.37112, 1.56763, 0.10145],
        ["H", -6.40989, 1.39069, -0.03899],
        ["H", -7.67217, 2.18471, -0.60766],
        ["H", -7.85266, 0.71353, -0.01396],
    ]

    # Write temporary files with the coordinates for water and oxonium and create
    # the corresponding systems
    tdir = mkdtemp(prefix="pka_workflow_", suffix=f"_tmp", dir=os.getcwd())

    with sh.pushd(tdir):
        with open("water.xyz", "w") as xyzfile:
            xyzfile.write(f"{len(WATER)}\n\n")
            for line in WATER:
                xyzfile.write("    ".join(str(x) for x in line) + "\n")

        with open("oxonium.xyz", "w") as xyzfile:
            xyzfile.write(f"{len(OXONIUM)}\n\n")
            for line in OXONIUM:
                xyzfile.write("    ".join(str(x) for x in line) + "\n")

        water = System("water.xyz", charge=0, spin=1)
        oxonium = System("oxonium.xyz", charge=1, spin=1)

    shutil.rmtree(tdir)

    # Create an empty dictionary to store the computed pKas and Gibbs free energies
    computed_pka, free_energies = {}, {}

    # Run geometry optimization for all structures in solvent
    protonated_sol = method_geometry.opt(protonated, ncores=ncores, maxcore=maxcore)
    deprotonated_sol = method_geometry.opt(deprotonated, ncores=ncores, maxcore=maxcore)
    water_sol = method_geometry.opt(water, ncores=ncores, maxcore=maxcore)
    oxonium_sol = method_geometry.opt(oxonium, ncores=ncores, maxcore=maxcore)

    # Run frequency calculation for all structures in solvent
    if method_energy != method_geometry:
        method_energy.freq(protonated_sol, ncores=ncores, maxcore=maxcore, inplace=True)
        method_energy.freq(deprotonated_sol, ncores=ncores, maxcore=maxcore, inplace=True)
        method_energy.freq(water_sol, ncores=ncores, maxcore=maxcore, inplace=True)
        method_energy.freq(oxonium_sol, ncores=ncores, maxcore=maxcore, inplace=True)
    
    # Extract the Gibbs Free Energies from the obtained systems
    G_protonated_sol = protonated_sol.properties.gibbs_free_energy
    G_deprotonated_sol = deprotonated_sol.properties.gibbs_free_energy
    G_water_sol = water_sol.properties.gibbs_free_energy
    G_oxonium_sol = oxonium_sol.properties.gibbs_free_energy

    # Store the computed Gibbs Free Energies for the calculations in solvent
    free_energies["G(solv) Protonated"] = G_protonated_sol
    free_energies["G(solv) Deprotonated"] = G_deprotonated_sol
    free_energies["G(solv) Water"] = G_water_sol
    free_energies["G(solv) Oxonium"] = G_oxonium_sol

    # Calculate the pKa using the direct scheme
    pKa_direct = calculate_pka(protonated_sol, deprotonated_sol)
    computed_pka["direct"] = pKa_direct

    # Calculate the pKa using the oxonium scheme
    pKa_oxonium = calculate_pka_oxonium_scheme(protonated_sol, deprotonated_sol, water_sol, oxonium_sol)
    computed_pka["oxonium"] = pKa_oxonium

    # If required run the cosmors based schemes
    if use_cosmors:

        # Run COSMO-RS calculation for all molecules
        dG_solv_prot = method_energy.cosmors(protonated_sol, ncores=ncores, maxcore=maxcore)
        dG_solv_deprot = method_energy.cosmors(deprotonated_sol, ncores=ncores, maxcore=maxcore)
        dG_solv_water = method_energy.cosmors(water_sol, ncores=ncores, maxcore=maxcore)
        dG_solv_oxonium = method_energy.cosmors(oxonium_sol, ncores=ncores, maxcore=maxcore)

        # Store the solvation free energies computed with OpenCOSMO-RS
        free_energies["dG(COSMO-RS) Protonated"] = dG_solv_prot
        free_energies["dG(COSMO-RS) Deprotonated"] = dG_solv_deprot
        free_energies["dG(COSMO-RS) Water"] = dG_solv_water
        free_energies["dG(COSMO-RS) Oxonium"] = dG_solv_oxonium

        # Create an engine for the calculations in vacuum
        method_energy_vac = deepcopy(method_energy)
        method_energy_vac.solvent = None

        # Run frequency calculation for all structures in vacuum
        protonated_vac = method_energy_vac.freq(protonated_sol, ncores=ncores, maxcore=maxcore)
        deprotonated_vac = method_energy_vac.freq(deprotonated_sol, ncores=ncores, maxcore=maxcore)
        water_vac = method_energy_vac.freq(water_sol, ncores=ncores, maxcore=maxcore)
        oxonium_vac = method_energy_vac.freq(oxonium_sol, ncores=ncores, maxcore=maxcore)

        # Extract the Gibbs Free Energies from the obtained systems
        G_protonated_vac = protonated_vac.properties.gibbs_free_energy
        G_deprotonated_vac = deprotonated_vac.properties.gibbs_free_energy
        G_water_vac = water_vac.properties.gibbs_free_energy
        G_oxonium_vac = oxonium_vac.properties.gibbs_free_energy

        # Store the Gibbs free energies computed in vacuum
        free_energies["G(vac) Protonated"] = protonated_vac.properties.gibbs_free_energy
        free_energies["G(vac) Deprotonated"] = deprotonated_vac.properties.gibbs_free_energy
        free_energies["G(vac) Water"] = water_vac.properties.gibbs_free_energy
        free_energies["G(vac) Oxonium"] = oxonium_vac.properties.gibbs_free_energy

        # Compute the reaction free energy in vacuum
        dG_vac = G_deprotonated_vac + G_oxonium_vac - G_protonated_vac - G_water_vac

        # Compute the difference in solvation free energy from OpenCOSMO-RS
        ddG_solv = dG_solv_deprot + dG_solv_oxonium - dG_solv_prot - dG_solv_water
        
        # Compute the corrected free energy in solvent and the corresponding pKa
        dG_cosmors = (dG_vac + ddG_solv) * scon.Eh_to_kcalmol
        pKa_cosmo = (dG_cosmors  / (np.log(10.0) * scon.R * 298.15 / scon.kcal_to_J)) - np.log10(997.0 / 18.01528)
        computed_pka["oxonium COSMO-RS"] = pKa_cosmo        

    return computed_pka, free_energies


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
