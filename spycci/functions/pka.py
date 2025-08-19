import logging
from copy import deepcopy
from typing import Union, Optional, Dict, Tuple

from spycci.systems import System
from spycci.core.base import Engine
from spycci.core.properties import pKa
from spycci.engines.xtb import XtbInput
from spycci.engines.orca import OrcaInput
from spycci.wrappers.crest import deprotonate
from spycci.tools.reorderenergies import reorder_energies
from spycci.functions.utils import retrieve_structure, validate_acid_base_pair, check_structure_acid_base_pair

import spycci.constants as scon
import numpy as np

logger = logging.getLogger(__name__)



def calculate_pka(protonated: System, deprotonated: System, only_return: bool = False) -> pKa:
    """
    Calculates the pKa of a molecule using the direct method. The function expects as arguments
    the protonated and deprotonated forms of the molecule as a `System` objects for which the
    energies must have already been computed. The function returns the computed pKa and sets (if
    not otherwise indicated by the user) the pka property of the `protonated` system.

    Parameters
    ----------
    protonated : System
        molecule in the protonated form
    deprotonated : System
        molecule in the deprotonated form
    only_return: bool
        If set to True will not set the computed pKa value in the properties of the `protonated`
        object. Else (default) the pKa property will be initialized using the `DIRECT` method.
    Returns
    -------
    pKa : pKa
        the pKa object encoding the pka of the molecule computed using the direct method.
    """
    logger.info("calculating pKa with direct scheme: HA -> A- + H+")

    # Validate systems and check if vibrational free energy corrections are available
    with_vibrations = validate_acid_base_pair(protonated, deprotonated)

    # Extract electronic energies
    protonated_energy = protonated.properties.electronic_energy
    deprotonated_energy = deprotonated.properties.electronic_energy

    # If available consider free energy correction factors from vibrational analysis
    free_energies = {}
    if with_vibrations:
        protonated_energy += protonated.properties.free_energy_correction
        deprotonated_energy += deprotonated.properties.free_energy_correction

        free_energies["G(solv) Protonated"] = protonated_energy
        free_energies["G(solv) Deprotonated"] = deprotonated_energy
    
    else:
        free_energies["Eel(solv) Protonated"] = protonated_energy
        free_energies["Eel(solv) Deprotonated"] = deprotonated_energy
    
    # Convert free energies (or electroni energies) from Hartree to kcal/mol
    protonated_energy *= scon.Eh_to_kcalmol
    deprotonated_energy *= scon.Eh_to_kcalmol

    # If gfn2 from xTB is used consider an additional correction factor for the proton self energy
    proton_self_energy = 0
    if "gfn2" in protonated.properties.level_of_theory_electronic:
        proton_self_energy = 164.22  # kcal/mol

    # Compute the pKa and set it as a property of the protonated molecule
    direct = (
        (
            deprotonated_energy
            + (scon.proton_hydration_free_energy + proton_self_energy)
            - protonated_energy
        )
    ) / (np.log(10.0) * scon.R * 298.15 / scon.kcal_to_J)

    pka = pKa()
    pka.set_direct(direct)
    pka.free_energies = free_energies
    
    if only_return is False:
        protonated.properties.set_pka(
            value=pka,
            electronic_engine=protonated.properties.level_of_theory_electronic,
            vibrational_engine=protonated.properties.level_of_theory_vibrational,
        )

    return pka


def calculate_pka_oxonium_scheme(
    protonated: System, deprotonated: System, water: System, oxonium: System, only_return: bool = False
) -> pKa:
    """
    Calculates the pKa of a molecule using the oxonium method. The function expects as arguments
    the protonated and deprotonated forms of the molecule and the water and oconium ion structures
    as a `System` objects for which the energies must have already been computed. The function 
    returns the computed pKa and sets (if not otherwise indicated by the user) the pka property of
    the `protonated` system.

    Parameters
    ----------
    protonated : System
        molecule in the protonated form
    deprotonated : System
        molecule in the deprotonated form
    water : System
        the water molecule
    oxonium: System
        the oxonium ion
    only_return: bool
        If set to True will not set the computed pKa value in the properties of the `protonated`
        object. Else (default) the pKa property will be initialized using the `DIRECT` method.

    Returns
    -------
    pKa : pKa
        the pKa object encoding the pka of the molecule computed using the direct method.
    """
    logger.info("calculating pKa with oxonium scheme: HA + H2O -> A- + H3O+")
    
    # Validate systems and check if vibrational free energy corrections are available
    with_vibrations = validate_acid_base_pair(protonated, deprotonated)

    # Extract electronic energies
    protonated_energy = protonated.properties.electronic_energy
    deprotonated_energy = deprotonated.properties.electronic_energy
    water_energy = water.properties.electronic_energy
    oxonium_energy = oxonium.properties.electronic_energy

    # If available consider free energy correction factors from vibrational calculations
    free_energies = {}
    if with_vibrations:
        protonated_energy += protonated.properties.free_energy_correction
        deprotonated_energy +=deprotonated.properties.free_energy_correction
        water_energy += water.properties.free_energy_correction
        oxonium_energy += oxonium.properties.free_energy_correction

        free_energies["G(solv) Protonated"] = protonated_energy
        free_energies["G(solv) Deprotonated"] = deprotonated_energy
        free_energies["G(solv) Water"] = water_energy
        free_energies["G(solv) Oxonium"] = oxonium_energy
    
    else:
        free_energies["Eel(solv) Protonated"] = protonated_energy
        free_energies["Eel(solv) Deprotonated"] = deprotonated_energy
        free_energies["Eel(solv) Water"] = water_energy
        free_energies["Eel(solv) Oxonium"] = oxonium_energy

    # Convert free energies (or electroni energies) from Hartree to kcal/mol
    protonated_energy *= scon.Eh_to_kcalmol
    deprotonated_energy *= scon.Eh_to_kcalmol
    water_energy *= scon.Eh_to_kcalmol
    oxonium_energy *= scon.Eh_to_kcalmol

    # Compute the pKa and set it as a property of the protonated molecule
    oxonium = (
        (deprotonated_energy + oxonium_energy - protonated_energy - water_energy)
    ) / (np.log(10.0) * scon.R * 298.15 / scon.kcal_to_J) - np.log10(997.0 / 18.01528)

    pka = pKa()
    pka.set_oxonium(oxonium)
    pka.free_energies = free_energies

    if only_return is False:
        protonated.properties.set_pka(
            value=pka,
            electronic_engine=protonated.properties.level_of_theory_electronic,
            vibrational_engine=protonated.properties.level_of_theory_vibrational,
        )

    return pka


def run_pka_workflow(
    protonated: System,
    deprotonated: System,
    method_vibrational: Union[XtbInput, OrcaInput],
    method_electonic: Optional[Union[XtbInput, OrcaInput]] = None,
    method_geometry: Optional[Union[XtbInput, OrcaInput]] = None,
    use_cosmors: bool = False,
    use_engine_settings: bool = False,
    ncores: Optional[int] = None,
    maxcore: int = 350,
) -> Tuple[pKa, System]:
    """
    The function runs a complete pKa workflow and retuns the pKa values, computed with different schemes, 
    and all the computed Gibbs free energies. The method runs a geometry optimization of both the protonated and
    deprotonated species in solvent (water). On the obtained geometries a frequecy calculation and, if necessaty,
    a single point calculation are carried out to compute the Gibbs free energies for all the species. 
    Water and Oxonium ion are automatically generated and optimized. The function then computes
    the pKa using the direct scheme and the oxonium scheme. If the `use_cosmors` option is used, the function
    also computes frequencies in vacuum and solvation free energies using OpenCOSMO-RS interface (requires orca>=6.0.0).
    The COSMO-RS calculation are then used to obtain the corresponding pKa using the oxonium method.

    Arguments
    ---------
    protonated : System
        The protonated system for which the pKa must be computed
    deprotonated : System
        The deprotomer generated during the dissociation reaction
    method_vibrational: Union[XtbInput, OrcaInput]
        The engine to be used to run the frequency calculations.
    method_electonic : Optional[Union[XtbInput, OrcaInput]]
        The engine to be used to run the electronic calculations. If set to `None` (default) will use electronic 
        energy computed by the `method_vibrational` engine. Please notice that, if the electronic method is different
        from the vibrational one, the computed Gibbs Free energy will be a mix of two different levels of theory (not
        advaisable)
    method_geometry : Optional[Union[XtbInput, OrcaInput]]
        The engine to be used to run the geometry optimizations. If set to `None` the user-provided geometries will be
        use directly without optimization while the water molecule and oxonium ion structures will be otimized using
        the BP86/def2-TZVPD (as the default OpenCOSMO-RS settings).
    use_cosmors : bool
        If set to `True` will also use OpenCOSMO-RS to compute solvation energies.
    use_engine_settings : bool
        If set to `True` will use the engine level of theory to run the COSMO-RS calculation (not advisable) else
        the default BP86/def2-TZVPD level of theory will be used.
    ncores : Optional[int]
        The number of cores to be used in the calculations. If set to `None` (default) will use
        the maximun number of available cores.
    maxcore: Optional[int]
        For the engines that supprots it, the memory assigned to each core used in the
        computation.

    Retruns
    -------
    pKa
        The pKa object containing all the computed pKa values and the Gibbs Free energies used in the computations.
    System
        The protonated system optimized in solvent in which the pKa property has been set.
    """
    logger.info("Running pKa workflow")

    # Check if the given structures are compatible with a pKa calculation
    check_structure_acid_base_pair(protonated, deprotonated)

    # Check if the method to be used for frequency calculations is xtb or orca
    if type(method_vibrational) not in [XtbInput, OrcaInput]:
        raise TypeError("The frequency calculation method must be xTB or orca.")
    
    # Check if the method to be used for energy calculations is xtb or orca
    if type(method_electonic) not in [XtbInput, OrcaInput] and method_electonic is not None:
        raise TypeError("The energy calculation method must be xTB or orca.")
    
    # Create an empty dictionary to store the computed pKas and Gibbs free energies
    free_energies = {}

    # Create a pKa class object to store all calculation results
    pka = pKa()
    
    # Define structures of water and oxonium ion
    water_xyz = retrieve_structure("water")
    water = System("water", charge=0, spin=1, geometry=water_xyz)

    oxonium_xyz = retrieve_structure("oxonium")
    oxonium = System("oxonium", charge=1, spin=1, geometry=oxonium_xyz)

    # RUN THE REQUIRED GEOMETRY OPTIMIZATIONS
    if method_geometry is None:
        logger.warning("No geometry method provided, the pKa calculations will run using the user-provided structures")
        logger.info("The geometries for the water and oxonium ions will be computed using BP86/def2-TZVPD")

        # Run only the geometry optimizations for the water and oxonium molecules using BP86/def2-TZVPD as default
        default_geom_engine = OrcaInput(method="BP86", basis_set="def2-TZVPD", solvent="water")
        water_sol = default_geom_engine.opt(water, ncores=ncores, maxcore=maxcore)
        oxonium_sol = default_geom_engine.opt(oxonium, ncores=ncores, maxcore=maxcore)

        # Copy user provided input structures
        protonated_sol = deepcopy(protonated)
        deprotonated_sol = deepcopy(deprotonated)

    elif type(method_geometry) not in [XtbInput, OrcaInput]:
        raise TypeError("The geometry optimization method must be xTB or orca.")

    else:
        # Run geometry optimization for all structures in solvent
        protonated_sol = method_geometry.opt(protonated, ncores=ncores, maxcore=maxcore)
        deprotonated_sol = method_geometry.opt(deprotonated, ncores=ncores, maxcore=maxcore)
        water_sol = method_geometry.opt(water, ncores=ncores, maxcore=maxcore)
        oxonium_sol = method_geometry.opt(oxonium, ncores=ncores, maxcore=maxcore)

    # Run frequency calculation for all structures in solvent
    if method_vibrational != method_geometry:
        method_vibrational.freq(protonated_sol, ncores=ncores, maxcore=maxcore, inplace=True)
        method_vibrational.freq(deprotonated_sol, ncores=ncores, maxcore=maxcore, inplace=True)
        method_vibrational.freq(water_sol, ncores=ncores, maxcore=maxcore, inplace=True)
        method_vibrational.freq(oxonium_sol, ncores=ncores, maxcore=maxcore, inplace=True)
    
    # If required run a single point calculation in solvent
    if method_electonic is not None and method_electonic != method_vibrational:
        method_electonic.spe(protonated_sol, ncores=ncores, maxcore=maxcore, inplace=True)
        method_electonic.spe(deprotonated_sol, ncores=ncores, maxcore=maxcore, inplace=True)
        method_electonic.spe(water_sol, ncores=ncores, maxcore=maxcore, inplace=True)
        method_electonic.spe(oxonium_sol, ncores=ncores, maxcore=maxcore, inplace=True)
    
    # Extract the Gibbs Free Energies from the obtained systems as E(el) + G-E(el)
    G_protonated_sol = protonated_sol.properties.electronic_energy
    G_deprotonated_sol = deprotonated_sol.properties.electronic_energy
    G_water_sol = water_sol.properties.electronic_energy
    G_oxonium_sol = oxonium_sol.properties.electronic_energy

    G_protonated_sol += protonated_sol.properties.free_energy_correction
    G_deprotonated_sol += deprotonated_sol.properties.free_energy_correction
    G_water_sol += water_sol.properties.free_energy_correction
    G_oxonium_sol += oxonium_sol.properties.free_energy_correction

    # Store the computed Gibbs Free Energies for the calculations in solvent
    free_energies["G(solv) Protonated"] = G_protonated_sol
    free_energies["G(solv) Deprotonated"] = G_deprotonated_sol
    free_energies["G(solv) Water"] = G_water_sol
    free_energies["G(solv) Oxonium"] = G_oxonium_sol

    # Calculate the pKa using the direct scheme
    pKa_direct = calculate_pka(protonated_sol, deprotonated_sol, only_return=True).direct
    pka.set_direct(pKa_direct)

    # Calculate the pKa using the oxonium scheme
    pKa_oxonium = calculate_pka_oxonium_scheme(protonated_sol, deprotonated_sol, water_sol, oxonium_sol, only_return=True).oxonium
    pka.set_oxonium(pKa_oxonium)

    # If required run the cosmors based schemes
    method_cosmors = method_electonic if method_electonic else method_vibrational
    if use_cosmors and type(method_cosmors)==OrcaInput:
        logger.info("Calculating pKa with COSMO-RS oxonium scheme: HA + H2O -> A- + H3O+")

        # Run COSMO-RS calculation for all molecules
        dG_solv_prot = method_cosmors.cosmors(protonated_sol,solvent="water", use_engine_settings=use_engine_settings, ncores=ncores, maxcore=maxcore)
        dG_solv_deprot = method_cosmors.cosmors(deprotonated_sol,solvent="water", use_engine_settings=use_engine_settings, ncores=ncores, maxcore=maxcore)
        dG_solv_water = method_cosmors.cosmors(water_sol,solvent="water", use_engine_settings=use_engine_settings, ncores=ncores, maxcore=maxcore)
        dG_solv_oxonium = method_cosmors.cosmors(oxonium_sol,solvent="water", use_engine_settings=use_engine_settings, ncores=ncores, maxcore=maxcore)

        # Store the solvation free energies computed with OpenCOSMO-RS
        free_energies["dG(COSMO-RS) Protonated"] = dG_solv_prot
        free_energies["dG(COSMO-RS) Deprotonated"] = dG_solv_deprot
        free_energies["dG(COSMO-RS) Water"] = dG_solv_water
        free_energies["dG(COSMO-RS) Oxonium"] = dG_solv_oxonium

        # Create an engine for the frequency calculations in vacuum
        method_vibrational_vac = deepcopy(method_vibrational)
        method_vibrational_vac.solvent = None

        # Run frequency calculation for all structures in vacuum
        protonated_vac = method_vibrational_vac.freq(protonated_sol, ncores=ncores, maxcore=maxcore)
        deprotonated_vac = method_vibrational_vac.freq(deprotonated_sol, ncores=ncores, maxcore=maxcore)
        water_vac = method_vibrational_vac.freq(water_sol, ncores=ncores, maxcore=maxcore)
        oxonium_vac = method_vibrational_vac.freq(oxonium_sol, ncores=ncores, maxcore=maxcore)

        # If required perform a single point calculation in vacuum
        if method_electonic is not None and method_electonic != method_vibrational:

            # Create an engine for the electronic calculations in vacuum
            method_electonic_vac = deepcopy(method_electonic)
            method_electonic_vac.solvent = None

            # Run frequency calculation for all structures in vacuum
            method_electonic_vac.spe(protonated_vac, ncores=ncores, maxcore=maxcore, inplace=True)
            method_electonic_vac.spe(deprotonated_vac, ncores=ncores, maxcore=maxcore, inplace=True)
            method_electonic_vac.spe(water_vac, ncores=ncores, maxcore=maxcore, inplace=True)
            method_electonic_vac.spe(oxonium_vac, ncores=ncores, maxcore=maxcore, inplace=True)

        # Extract the Gibbs Free Energies from the obtained systems as E(el) + G-E(el)
        G_protonated_vac = protonated_vac.properties.electronic_energy
        G_deprotonated_vac = deprotonated_vac.properties.electronic_energy
        G_water_vac = water_vac.properties.electronic_energy
        G_oxonium_vac = oxonium_vac.properties.electronic_energy

        G_protonated_vac += protonated_vac.properties.free_energy_correction
        G_deprotonated_vac += deprotonated_vac.properties.free_energy_correction
        G_water_vac += water_vac.properties.free_energy_correction
        G_oxonium_vac += oxonium_vac.properties.free_energy_correction

        # Store the Gibbs free energies computed in vacuum
        free_energies["G(vac) Protonated"] = G_protonated_vac
        free_energies["G(vac) Deprotonated"] = G_deprotonated_vac
        free_energies["G(vac) Water"] = G_water_vac
        free_energies["G(vac) Oxonium"] = G_oxonium_vac

        # Compute the reaction free energy in vacuum
        dG_vac = G_deprotonated_vac + G_oxonium_vac - G_protonated_vac - G_water_vac

        # Compute the difference in solvation free energy from OpenCOSMO-RS
        ddG_solv = dG_solv_deprot + dG_solv_oxonium - dG_solv_prot - dG_solv_water
        
        # Compute the corrected free energy in solvent and the corresponding pKa
        dG_cosmors = (dG_vac + ddG_solv) * scon.Eh_to_kcalmol
        pKa_cosmo = (dG_cosmors  / (np.log(10.0) * scon.R * 298.15 / scon.kcal_to_J)) - np.log10(997.0 / 18.01528)
        pka.set_oxonium_cormors(pKa_cosmo, method_cosmors)

    elif use_cosmors:
            logger.warning("COSMO-RS calculations can be run only by OrcaInput engine as electronic method.")    

    pka.free_energies = free_energies



    protonated_sol.properties.set_pka(
        pka,
        electronic_engine=method_electonic if method_electonic else method_vibrational,
        vibrational_engine=method_vibrational,
    )

    return pka, protonated_sol


def auto_calculate_pka(
    protonated: System,
    method_el: Engine,
    method_vib: Engine = None,
    method_opt: Engine = None,
    ncores: int = None,
    maxcore: int = 350,
) -> Tuple[pKa, System]:
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
        The computational engine to be used in the vibrational level of theory calculations. If
        set to None (default) the pKa will be computed without the free energy corrections (only electronic energy).
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
    pKa
        pKa object computed for the molecule.
    System
        the structure of the considered deprotomer.
    """
    logger.info("Running auto-calculate pKa")
    
    if method_opt is None:
        method_opt = XtbInput(solvent="water")

    method_opt.opt(protonated, inplace=True, ncores=ncores, maxcore=maxcore)

    if method_vib == method_el:
        method_el.freq(protonated, inplace=True, ncores=ncores, maxcore=maxcore)
    else:
        method_el.spe(protonated, inplace=True, ncores=ncores, maxcore=maxcore)

        if method_vib is not None and method_vib != method_opt:
            dummy: System = method_vib.freq(protonated, ncores=ncores, maxcore=maxcore)
            protonated.properties.set_free_energy_correction(
                dummy.properties.free_energy_correction, method_vib
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
