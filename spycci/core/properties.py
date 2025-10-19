from __future__ import annotations

import logging
import spycci.config

from copy import deepcopy
from typing import Dict, List, Union, Callable, TYPE_CHECKING
from spycci.config import StrictnessLevel
from spycci.core.base import Engine
from spycci.core.spectroscopy import VibrationalData

if TYPE_CHECKING:
    from spycci.systems import System

logger = logging.getLogger(__name__)


def is_orca_level_of_theory(string: str) -> bool:
    """
    Checks whether the input string matches the standard for the level of theory string in
    OrcaInput:

    "OrcaInput || method: § | basis: § | solvent: §"

    Parameters
    ----------
    string: str
        String to be checked

    Returns
    -------
    bool
        True if the string matches the standard OrcaInput level of theory format, else False
    """
    for required in ["OrcaInput || method: ", " | basis: ", " | solvent: "]:
        if required not in string:
            return False

    return True


def is_xtb_level_of_theory(string: str) -> bool:
    """
    Checks whether the input string matches the standard for the level of theory string in
    XtbInput:

    "XtbInput || method: § | solvent: §"

    Parameters
    ----------
    string: str
        String to be checked

    Returns
    -------
    bool
        True if the string matches the standard XtbInput level of theory format, else False
    """
    for required in ["XtbInput || method: ", " | solvent: "]:
        if required not in string:
            return False

    return True


def is_dftb_level_of_theory(string: str) -> bool:
    """
    Checks whether the input string matches the standard for the level of theory string in
    DFTBInput:

    "DFTBInput || method: § | parameters: § | 3rd order: § | dispersion: §"

    Parameters
    ----------
    string: str
        String to be checked

    Returns
    -------
    bool
        True if the string matches the standard DFTBInput level of theory format, else False
    """
    for required in [
        "DFTBInput || method: ",
        " | parameters: ",
        " | 3rd order: ",
        " | dispersion: ",
    ]:
        if required not in string:
            return False

    return True


class Properties:
    """
    Class containing the properties associated to a system when a given electronic and vibrational
    level of theory are considered. The class automatically stores the level of theory associated
    to the coputed properties and checks, whenever a new property is set, that a given input data
    is compatible with the current used level of theory. If a mismatch between levels of theory is
    detected all the properties related to the old level of theory are cleaned and a warning is
    raised. The level of sanity check applied to the levels of theory is set by the `STRICTNESS_LEVEL`
    variable of the `spycci.config` module. In `NORMAL` mode the electronic and vibrational levels
    of theory can be different while in `STRICT` equality is enforced to ensure consistency among
    electronic and vibrational levels of theory. In `STRICT` mode a change of a level of theory will,
    in case of a mismatch, clear the properties associated to the other. In cased of mixed properties
    (e.g. pKa in which electronic and vibronic levels of theory are set simultaneously) exception is
    raised in `STRICT` mode when mismatch is detected.
    """

    def __init__(self):
        self.__level_of_theory_electronic: str = None
        self.__level_of_theory_vibrational: str = None

        self.__electronic_energy: float = None
        self.__free_energy_correction: float = None
        self.__pka: pKa = pKa()
        self.__mulliken_charges: List[float] = []
        self.__mulliken_spin_populations: List[float] = []
        self.__condensed_fukui_mulliken: Dict[str, List[float]] = {}
        self.__hirshfeld_charges: List[float] = []
        self.__hirshfeld_spin_populations: List[float] = []
        self.__condensed_fukui_hirshfeld: Dict[str, List[float]] = {}
        self.__vibrational_data: VibrationalData = None

        # Define a listener to validate geometry level of theory stored in System
        self.__check_geometry_level_of_theory: System.__check_geometry_level_of_theory = None
    
    def __add_check_geometry_level_of_theory(self, listener: System.__check_geometry_level_of_theory) -> None:
        """
        Add a reference to a `System` function accepting as argument the geometry
        level of theory to be checked. (implemented as `__check_geometry_level_of_theory`)

        Argument
        --------
        listener: System.__check_geometry_level_of_theory
            The method `__check_geometry_level_of_theory` of the `System` handling the level of theory check
        """
        self.__check_geometry_level_of_theory = listener
    
    def __call_check_geometry_level_of_theory(self, level_of_theory: str) -> None:
        "If set, send to the system (owner) the engine to be checked"
        if self.__check_geometry_level_of_theory is not None:
            self.__check_geometry_level_of_theory(level_of_theory)
    
    def __deepcopy__(self, memo) -> Properties:
        "Overload of the deepcopy funtion to safely remove listener reference"
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj

        for attr_name, attr_value in self.__dict__.items():
            setattr(obj, attr_name, deepcopy(attr_value, memo))
        
        obj.__check_geometry_level_of_theory = None

        return obj

    def __clear_electronic(self):
        self.__level_of_theory_electronic = None
        self.__electronic_energy = None
        self.__pka = pKa()
        self.__mulliken_charges = []
        self.__mulliken_spin_populations = []
        self.__condensed_fukui_mulliken = {}
        self.__hirshfeld_charges = []
        self.__hirshfeld_spin_populations = []
        self.__condensed_fukui_hirshfeld = {}

    def __clear_vibrational(self):
        self.__level_of_theory_vibrational = None
        self.__free_energy_correction = None
        self.__pka = pKa()
        self.__vibrational_data = None

    def __check_engine(self, engine: Union[Engine, str]) -> None:
        
        level_of_theory = None
        logger.debug(f"Engine type: {type(engine)}")
        
        if type(engine) == str:
            if not any(
                [
                    is_orca_level_of_theory(engine),
                    is_xtb_level_of_theory(engine),
                    is_dftb_level_of_theory(engine),
                ]
            ):
                raise TypeError(
                    "The engine argument string does not match any valid level of theory"
                )
            else:
                level_of_theory = engine

        elif isinstance(engine, Engine):
            level_of_theory = engine.level_of_theory

        else:
            raise TypeError("The engine argument must be derived from `Engine`")
        
        if spycci.config.STRICTNESS_LEVEL == spycci.config.StrictnessLevel.VERY_STRICT:
            self.__call_check_geometry_level_of_theory(level_of_theory)
        
        return level_of_theory
        

    def __validate_electronic(self, engine: Union[Engine, str]) -> None:

        level_of_theory = self.__check_engine(engine)

        logger.debug("VALIDATING ELECTRONIC LEVEL OF THEORY")
        logger.debug(f"Strictness level: {spycci.config.STRICTNESS_LEVEL.name}")
        logger.debug(f"Current electronic level of theory: {self.__level_of_theory_electronic}")
        logger.debug(f"Current vibrational level of theory: {self.__level_of_theory_vibrational}")
        logger.debug(f"Requested electronic level of theory: {level_of_theory}")

        if self.__level_of_theory_electronic is None:
            self.__level_of_theory_electronic = level_of_theory

        elif self.__level_of_theory_electronic != level_of_theory:
            msg = "Different electronic levels of theory used for calculating properties. Clearing properties with different electronic level of theory."
            logger.warning(msg)
            self.__clear_electronic()
            self.__level_of_theory_electronic = level_of_theory

        if self.__level_of_theory_vibrational and self.__level_of_theory_electronic != self.__level_of_theory_vibrational:  
            if spycci.config.STRICTNESS_LEVEL in [StrictnessLevel.STRICT]:
                    msg = "The electronic and vibrational levels of theory differs. Clearing properties with different vibrational level of theory."
                    logger.warning(msg)
                    self.__clear_vibrational()

    def __validate_vibrational(self, engine: Engine) -> None:

        level_of_theory = self.__check_engine(engine)

        logger.debug("VALIDATING VIBRATIONAL LEVEL OF THEORY")
        logger.debug(f"Strictness level: {spycci.config.STRICTNESS_LEVEL.name}")
        logger.debug(f"Current electronic level of theory: {self.__level_of_theory_electronic}")
        logger.debug(f"Current vibrational level of theory: {self.__level_of_theory_vibrational}")
        logger.debug(f"Requested vibrational level of theory: {level_of_theory}")

        if self.__level_of_theory_vibrational is None:
            
            if self.__pka.is_set() is True:
                msg = "Added vibrational contribution. Clearing pKa computed with electronic energy only."
                logger.warning(msg)
                self.__pka = pKa()

            self.__level_of_theory_vibrational = level_of_theory

        elif self.__level_of_theory_vibrational != level_of_theory:
            msg = "Different vibrational levels of theory used for calculating properties. Clearing properties with different vibrational level of theory."
            logger.warning(msg)
            self.__clear_vibrational()
            self.__level_of_theory_vibrational = level_of_theory
        
        if self.__level_of_theory_electronic and self.__level_of_theory_electronic != self.__level_of_theory_vibrational:  
            if spycci.config.STRICTNESS_LEVEL in [StrictnessLevel.STRICT]:
                    msg = "The electronic and vibrational levels of theory differs. Clearing properties with different electronic level of theory."
                    logger.warning(msg)
                    self.__clear_electronic()
    
    def __validate_strictness_simultaneously(self, el_engine: Engine, vib_engine: Engine) -> None:

        el_level_of_theory = self.__check_engine(el_engine)
        vib_level_of_theory = self.__check_engine(vib_engine)

        if spycci.config.STRICTNESS_LEVEL in [StrictnessLevel.STRICT]:
            if el_level_of_theory != vib_level_of_theory:
                msg = f"Mismatch between levels of theory of composite property detected during setting in {spycci.config.STRICTNESS_LEVEL.name} mode."
                logger.error(msg)
                raise RuntimeError(msg)


    def to_dict(self) -> dict:
        """
        Generates a dictionary representation of the class. The obtained dictionary can be
        saved and used to re-load the object using the built-in `from_dict` class method.

        Returns
        -------
        dict
            The dictionary listing, with human friendly names, the attributes of the class
        """
        data = {}
        data["Level of theory electronic"] = self.__level_of_theory_electronic
        data["Level of theory vibrational"] = self.__level_of_theory_vibrational
        data["Electronic energy (Eh)"] = self.__electronic_energy
        data["Free energy correction G-E(el) (Eh)"] = self.__free_energy_correction
        data["pKa"] = self.__pka.to_dict()
        data["Mulliken charges"] = self.__mulliken_charges
        data["Mulliken spin populations"] = self.__mulliken_spin_populations
        data["Mulliken Fukui"] = self.__condensed_fukui_mulliken
        data["Hirshfeld charges"] = self.__hirshfeld_charges
        data["Hirshfeld spin populations"] = self.__hirshfeld_spin_populations
        data["Hirshfeld Fukui"] = self.__condensed_fukui_hirshfeld

        if self.__vibrational_data is None:
            data["Vibrational data"] = None
        else:
            data["Vibrational data"] = self.__vibrational_data.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict) -> Properties:
        """
        Construct a Properties object from the data encoded in a dictionary.

        Arguments
        ---------
        data: dict
            The dictionary containing the class attributes

        Returns
        -------
        Properties
            The fully initialized Properties object
        """
        obj = cls()
        obj.__level_of_theory_electronic = data["Level of theory electronic"]
        obj.__level_of_theory_vibrational = data["Level of theory vibrational"]
        obj.__electronic_energy = data["Electronic energy (Eh)"]
        obj.__free_energy_correction = data["Free energy correction G-E(el) (Eh)"]
        obj.__pka.from_dict(data["pKa"])
        obj.__mulliken_charges = data["Mulliken charges"]
        obj.__mulliken_spin_populations = data["Mulliken spin populations"]
        obj.__condensed_fukui_mulliken = data["Mulliken Fukui"]
        obj.__hirshfeld_charges = data["Hirshfeld charges"]
        obj.__hirshfeld_spin_populations = data["Hirshfeld spin populations"]
        obj.__condensed_fukui_hirshfeld = data["Hirshfeld Fukui"]

        if data["Vibrational data"] is None:
            obj.__vibrational_data = None
        else:
            obj.__vibrational_data = VibrationalData.from_dict(data["Vibrational data"])

        return obj

    @property
    def level_of_theory_electronic(self) -> str:
        """
        The level of theory adopted for the electronic structure calculations

        Returns
        -------
        str
            The string encoding the electronic level of theory
        """
        return self.__level_of_theory_electronic

    @property
    def level_of_theory_vibrational(self) -> str:
        """
        The level of theory adopted for the vibrational calculations

        Returns
        -------
        str
            The string encoding the vibrational level of theory
        """
        return self.__level_of_theory_vibrational

    @property
    def electronic_energy(self) -> float:
        """
        The electronic energy of the system in Hartree.

        Returns
        -------
        float
            The electronic energy of the system in Hartree.
        """
        return self.__electronic_energy

    def set_electronic_energy(self, value: float, electronic_engine: Union[Engine, str]) -> None:
        """
        Sets the electronic energy of the system.

        Arguments
        ---------
        value: float
            The electronic energy of the system in Hartree.
        electronic_engine: Union[Engine, str]
            The engine used in the calculation.
        """
        logger.debug("Setting electronic energy")
        self.__validate_electronic(electronic_engine)
        self.__electronic_energy = value

    @property
    def free_energy_correction(self) -> float:
        """
        The free energy correction G-E(el) as the difference between the Gibbs free energy
        and the electronic energy (in Hartree).

        Returns
        -------
        float
            The free energy correction in Hartree.
        """
        return self.__free_energy_correction

    def set_free_energy_correction(self, value: float, vibrational_engine: Union[Engine, str]) -> None:
        """
        Sets the free energy correction G-E(el) of the system.

        Arguments
        ---------
        value: float
            The free energy correction of the system in Hartree.
        vibrational_engine: Union[Engine, str]
            The engine used in the calculation.
        """
        logger.debug("Setting Gibbs free energy correction")
        self.__validate_vibrational(vibrational_engine)
        self.__free_energy_correction = value

    @property
    def gibbs_free_energy(self) -> float:
        """
        The Gibbs free energy of the system in Hartree as the
        sum of Electronic energy and free energy correction

        Returns
        -------
        float
            The Gibbs free energy of the system in Hartree.
        """
        if self.__electronic_energy and self.__free_energy_correction:
            if self.level_of_theory_electronic != self.level_of_theory_vibrational:
                logger.warning("Gibbs free energy has been computed mixing different levels of theory")
            return self.__electronic_energy + self.__free_energy_correction
        else:
            return None

    @property
    def pka(self) -> pKa:
        """
        The pKa of the system.

        Returns
        -------
        float
            The pKa the system.
        """
        return self.__pka

    def set_pka(
        self,
        value: pKa,
        electronic_engine: Union[Engine, str],
        vibrational_engine: Union[Engine, str] = None,
    ) -> None:
        """
        Sets the pKa of the system.

        Arguments
        ---------
        value: float
            The pKa of the system.
        electronic_engine: Union[Engine, str]
            The engine used in the electronic calculation.
        vibrational_engine: Union[Engine, str]
            The engine used in the vibrational calculation. (optional)
        
        Raises
        ------
        RuntimeError
            Exception raised if a mismatch between levels of theory is detected in STRICT mode.
        """
        logger.debug("Setting pKa")

        if vibrational_engine is not None:
            self.__validate_strictness_simultaneously(electronic_engine, vibrational_engine)

        self.__validate_electronic(electronic_engine)
        if vibrational_engine is not None:
            self.__validate_vibrational(vibrational_engine)

        if type(value) != pKa:
            raise TypeError("The pka value must be of pKa class object type.")
        self.__pka = value

    @property
    def mulliken_charges(self) -> List[float]:
        """
        The Mulliken charges of the system.

        Returns
        -------
        List[float]
            The list of Mulliken charges associated to each atom in the system.
        """
        return self.__mulliken_charges

    def set_mulliken_charges(self, value: List[float], electronic_engine: Union[Engine, str]) -> None:
        """
        Sets the Mulliken charges of the system.

        Arguments
        ---------
        value: float
            The list of Mulliken charges associated to each atom of the system.
        electronic_engine: Union[Engine, str]
            The engine used in the electronic calculation.
        """
        logger.debug("Setting Mulliken charges")
        self.__validate_electronic(electronic_engine)
        self.__mulliken_charges = value

    @property
    def mulliken_spin_populations(self) -> List[float]:
        """
        The Mulliken spin populations of the system.

        Returns
        -------
        List[float]
            The list of Mulliken spin populations associated to each atom in the system.
        """
        return self.__mulliken_spin_populations

    def set_mulliken_spin_populations(self, value: List[float], electronic_engine: Union[Engine, str]) -> None:
        """
        Sets the Mulliken spin populations of the system.

        Arguments
        ---------
        value: float
            The list of Mulliken spin populations associated to each atom of the system.
        electronic_engine: Union[Engine, str]
            The engine used in the electronic calculation.
        """
        logger.debug("Setting Mulliken Spin populations")
        self.__validate_electronic(electronic_engine)
        self.__mulliken_spin_populations = value

    @property
    def condensed_fukui_mulliken(self) -> Dict[str, List[float]]:
        """
        The condensed Fukui values computed from the Mulliken charges.

        Returns
        -------
        Dict[str, List[float]]
            The dictionaty containing the list of condensed Fukui values computed for each
            atom in the system starting from the values of the Mulliken charges. The functions
            are stored in the dictionary according to the `f+`, `f-` and `f0` keys.
        """
        return self.__condensed_fukui_mulliken

    def set_condensed_fukui_mulliken(
        self, value: Dict[str, List[float]], electronic_engine: Union[Engine, str]
    ) -> None:
        """
        Sets condensed Fukui values computed from the Mulliken charges.

        Arguments
        ---------
        Dict[str, List[float]]
            The dictionaty containing the list of condensed Fukui values computed for each
            atom in the system starting from the values of the Mulliken charges. The functions
            are stored in the dictionary according to the `f+`, `f-` and `f0` keys.
        electronic_engine: Union[Engine, str]
            The engine used in the electronic calculation.
        """
        logger.debug("Setting condensed Fukui functions (Mulliken)")
        self.__validate_electronic(electronic_engine)
        self.__condensed_fukui_mulliken = value

    @property
    def hirshfeld_charges(self) -> List[float]:
        """
        The Hirshfeld charges of the system.

        Returns
        -------
        List[float]
            The list of Hirshfeld charges associated to each atom in the system.
        """
        return self.__hirshfeld_charges

    def set_hirshfeld_charges(self, value: List[float], electronic_engine: Union[Engine, str]) -> None:
        """
        Sets the Hirshfeld charges of the system.

        Arguments
        ---------
        value: float
            The list of Hirshfeld charges associated to each atom of the system.
        electronic_engine: Union[Engine, str]
            The engine used in the electronic calculation.
        """
        logger.debug("Setting Hirshfeld charges")
        self.__validate_electronic(electronic_engine)
        self.__hirshfeld_charges = value

    @property
    def hirshfeld_spin_populations(self) -> List[float]:
        """
        The Hirshfeld spin populations of the system.

        Returns
        -------
        List[float]
            The list of Hirshfeld spin populations associated to each atom in the system.
        """
        return self.__hirshfeld_spin_populations

    def set_hirshfeld_spin_populations(self, value: List[float], electronic_engine: Union[Engine, str]) -> None:
        """
        Sets the Hirshfeld spin populations of the system.

        Arguments
        ---------
        value: float
            The list of Hirshfeld spin populations associated to each atom of the system.
        electronic_engine: Union[Engine, str]
            The engine used in the electronic calculation.
        """
        logger.debug("Setting Hirshfeld Spin populations")
        self.__validate_electronic(electronic_engine)
        self.__hirshfeld_spin_populations = value

    @property
    def condensed_fukui_hirshfeld(self) -> Dict[str, List[float]]:
        """
        The condensed Fukui values computed from the Hirshfeld charges.

        Returns
        -------
        Dict[str, List[float]]
            The dictionaty containing the list of condensed Fukui values computed for each
            atom in the system starting from the values of the Hirshfeld charges. The functions
            are stored in the dictionary according to the `f+`, `f-` and `f0` keys.
        """
        return self.__condensed_fukui_hirshfeld

    def set_condensed_fukui_hirshfeld(
        self, value: Dict[str, List[float]], electronic_engine: Union[Engine, str]
    ) -> None:
        """
        Sets condensed Fukui values computed from the Hirshfeld charges.

        Arguments
        ---------
        Dict[str, List[float]]
            The dictionaty containing the list of condensed Fukui values computed for each
            atom in the system starting from the values of the Hirshfeld charges. The functions
            are stored in the dictionary according to the `f+`, `f-` and `f0` keys.
        electronic_engine: Union[Engine, str]
            The engine used in the electronic calculation.
        """
        logger.debug("Setting condensed Fukui functions (Hirshfeld)")
        self.__validate_electronic(electronic_engine)
        self.__condensed_fukui_hirshfeld = value

    @property
    def vibrational_data(self) -> VibrationalData:
        """
        Returns the class containing all the available vibrational data about the molecule

        Returns
        -------
        VibrationalData
            The class containing all the available vibrational data
        """
        return self.__vibrational_data

    def set_vibrational_data(
        self,
        value: VibrationalData,
        vibrational_engine: Union[Engine, str],
    ) -> None:
        """
        Sets condensed Fukui values computed from the Hirshfeld charges.

        Arguments
        ---------
        value: VibrationalData
            The class encoding all the vibrational data associated to the molecule
        vibrational_engine: Union[Engine, str]
            The engine used in the vibrational calculation.
        """
        logger.debug("Setting vibrational data")
        self.__validate_vibrational(vibrational_engine)
        self.__vibrational_data = value


# Define the pKa property class to store multiple pKa values computed with different schemes
class pKa:
    """
    The pKa class organizes all the pKa values computed for a given System using different schemes.
    All the stored data are accessible as read-only properties (no setter is provided). To set the
    pKa values, dedicated set function are provided (e.g. `set_direct`). The class also exposes the
    `free_energies` attribute where all the free energy values used in the calculations are stored.

    Properties
    ----------
    direct: float
        The pKa value computed using the direct scheme.
    oxonium: float
        The pKa value computed using the oxonium scheme.
    oxonium_cosmors: float
        The pKa value computed using the COSMO-RS model to compute the solvation energies in the
        oxonium scheme
    level_of_theory_cosmors: str
        The level of theory used to run the COSMO-RS calculations
    """

    def __init__(self):
        self.__direct: float = None
        self.__oxonium: float = None
        self.__oxonium_cosmors: float = None

        self.free_energies: Dict[str, float] = None
        self.__level_of_theory_cosmors: str = None

    def __getitem__(self, key: str) -> Union[float, None]:
        if key.upper() == "DIRECT":
            return self.direct
        elif key.upper() == "OXONIUM":
            return self.oxonium
        elif key.upper() == "OXONIUM COSMO-RS":
            return self.oxonium_cosmors
        else:
            raise ValueError(f"The key {key} is not a valid pka scheme.")
    
    def __str__(self) -> str:

        if self.is_set() is False:
            return "pKa object status is NOT SET\n"

        string = ""
        
        if self.direct is not None:
            string += f"pKa direct: {self.direct}\n"
        if self.oxonium is not None:
            string += f"pKa oxonium: {self.oxonium}\n"
        if self.oxonium_cosmors is not None:
            string += f"pKa oxonium COSMO-RS: {self.oxonium_cosmors}\n"
            string += f"COSMO-RS level of theory: {self.level_of_theory_cosmors}\n"

        return string

    def is_set(self) -> bool:
        """
        Returns `True` if at least one value of the class has been set, `False` otherwise.

        Returns
        -------
        bool
            The set status of the pKa class object.
        """
        checklist = [
            self.__direct,
            self.__oxonium,
            self.__oxonium_cosmors,
            self.free_energies,
            self.__level_of_theory_cosmors,
        ]
        if any(x is not None for x in checklist):
            return True
        else:
            return False

    def to_dict(self) -> dict:
        """
        Generates a dictionary representation of the class. The obtained dictionary can be
        saved and used to re-load the object using the built-in `from_dict` class method.

        Returns
        -------
        dict
            The dictionary listing, with human friendly names, the attributes of the class
        """
        data = {}
        data["direct"] = self.__direct
        data["oxonium"] = self.__oxonium
        data["oxonium COSMO-RS"] = self.__oxonium_cosmors
        data["free energies"] = self.free_energies
        data["level of theory cosmors"] = self.__level_of_theory_cosmors
        return data

    @classmethod
    def from_dict(cls, data: dict) -> pKa:
        """
        Construct a pKa object from the data encoded in a dictionary.

        Arguments
        ---------
        data: dict
            The dictionary containing the class attributes

        Returns
        -------
        pKa
            The fully initialized pKa object
        """
        obj = cls()
        obj.__direct = data["direct"]
        obj.__oxonium = data["oxonium"]
        obj.__oxonium_cosmors = data["oxonium COSMO-RS"]
        obj.free_energies = data["free energies"]
        obj.__level_of_theory_cosmors = data["level of theory cosmors"]
        return obj

    @property
    def direct(self) -> Union[float, None]:
        r"""
        The pKa value computed using the direct scheme. According to the direct scheme,
        the pKa is computed from the standard thermodynamic dissociation reaction:

        

        The scheme is semi-empiric: the Gibbs free energies of HA and A⁻ are computed in 
        acqueous solvent while the proton free energy, together with the RTln(24.46), is
        assumed to be -270.29 kcal/mol at 298.15K.

        Returns
        -------
        Union[float, None]
            The pKa value as a float value, if available, else `None`.
        """
        return self.__direct

    def set_direct(self, pka: float) -> None:
        """
        Sets the pKa computed using the direct scheme.

        Arguments
        ---------
        pka: float
            The computed pKa value.
        """
        self.__direct = pka

    @property
    def oxonium(self) -> Union[float, None]:
        r"""
        The pKa value computed using the oxonium scheme. According to the oxonium scheme,
        the pKa is computed from the standard thermodynamic dissociation reaction:

        :math:`HA + H_2O \rightarrow H_3O^{+} + A^{-}`

        The scheme is fully computational: all species involved (HA, A⁻, H₂O, H₃O⁺)
        must have their free energies computed in aqueous solvent. The water concentration
        is taken as 55.34 mol/L (i.e., 997 g/L at 25°C / 18.01528 g/mol).

        Returns
        -------
        Union[float, None]
            The pKa value.
        """
        return self.__oxonium

    def set_oxonium(self, pka: float) -> None:
        """
        Sets the pKa computed using the oxonium scheme.

        Arguments
        ---------
        pka: float
            The computed pKa value.
        """
        self.__oxonium = pka

    @property
    def oxonium_cosmors(self) -> Union[float, None]:
        r"""
        The pKa value computed using the oxonium scheme and the COSMO-RS solvation energies.
        This variant of the oxonium scheme uses solvation free energies from a COSMO-RS
        calculation instead of implicit solvent methods. The thermodynamic cycle remains:

        :math:`HA + H_2O \rightarrow H_3O^{+} + A^{-}`

        The scheme is fully computational: all species involved (HA, A⁻, H₂O, H₃O⁺)
        must have their free energies computed according to an hybrid scheme. Firstly
        the free energy in vacumm is computed for all species (using the solvent equilibium
        geometyr). Then, the solvation correction for the water solvent is then taken from
        the OpenCOSMO-RS model. The water concentration is taken as 55.34 mol/L (i.e., 997 g/L
        at 25°C / 18.01528 g/mol).

        Returns
        -------
        Union[float, None]
            The pKa value.
        """
        return self.__oxonium_cosmors

    @property
    def level_of_theory_cosmors(self) -> Union[str, None]:
        """
        The level of theory used in the COSMO-RS calculations.

        Returns
        -------
        Union[str, None]
            The string encoding the level of theory.
        """
        return self.__level_of_theory_cosmors

    def set_oxonium_cormors(self, pka: float, engine: Engine) -> None:
        """
        Sets the pKa computed using the oxonium scheme using the solvation energies
        computing the COSMO-RS method.

        Arguments
        ---------
        pka: float
            The computed pKa value.
        engine: Engine
            The engine used in the COSMO-RS calculations.
        """
        if isinstance(engine, Engine) is False:
            raise TypeError("COSMO-RS level of theory must be of type engine")
        self.__oxonium_cosmors = pka
        self.__level_of_theory_cosmors = engine.level_of_theory
