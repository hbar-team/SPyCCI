from __future__ import annotations

import logging, warnings
import spycci.config
import numpy as np

from typing import Dict, List, Union
from spycci.core.base import Engine
from spycci.core.spectroscopy import VibrationalData


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
    is compatible with the current used level of theory. In normal conditions (strict mode)
    if a mismatch between levels of theory is detected all the properties related to the old
    level of theory are cleaned and a warning is raised. If the strict mode is disables
    (by setting compechem.config.STRICT_MODE to False) the level of theory is set to `Undefined`
    and all the properties are kept.
    """

    def __init__(self):
        self.__level_of_theory_electronic: str = None
        self.__level_of_theory_vibrational: str = None

        self.__electronic_energy: float = None
        self.__free_energy_correction: float = None
        self.__helmholtz_free_energy: float = None
        self.__pka: float = None
        self.__mulliken_charges: List[float] = []
        self.__mulliken_spin_populations: List[float] = []
        self.__condensed_fukui_mulliken: Dict[str, List[float]] = {}
        self.__hirshfeld_charges: List[float] = []
        self.__hirshfeld_spin_populations: List[float] = []
        self.__condensed_fukui_hirshfeld: Dict[str, List[float]] = {}
        self.__vibrational_data: VibrationalData = None

    def __clear_electronic(self):
        self.__level_of_theory_electronic = None
        self.__electronic_energy = None
        self.__helmholtz_free_energy = None
        self.__pka = None
        self.__mulliken_charges = []
        self.__mulliken_spin_populations = []
        self.__condensed_fukui_mulliken = {}
        self.__hirshfeld_charges = []
        self.__hirshfeld_spin_populations = []
        self.__condensed_fukui_hirshfeld = {}

    def __clear_vibrational(self):
        self.__level_of_theory_vibrational = None
        self.__free_energy_correction = None
        self.__helmholtz_free_energy = None
        self.__pka = None
        self.__vibrational_data = None

    def __check_engine(self, engine: Union[Engine, str]) -> None:

        logger.debug(f"Engine type: {type(engine)}")

        if type(engine) == str:
            if not any(
                [
                    is_orca_level_of_theory(engine),
                    is_xtb_level_of_theory(engine),
                    is_dftb_level_of_theory(engine),
                ]
            ):
                raise TypeError("The engine argument string does not match any valid level of theory")
            else:
                return engine

        elif isinstance(engine, Engine):
            return engine.level_of_theory

        else:
            raise TypeError("The engine argument must be derived from `Engine`")

    def __validate_electronic(self, engine: Union[Engine, str]) -> None:

        level_of_theory = self.__check_engine(engine)

        logger.debug("Validating electronic energy")
        logger.debug(f"current: {self.__level_of_theory_electronic}, requested: {level_of_theory}")

        if self.__level_of_theory_electronic is None:
            self.__level_of_theory_electronic = level_of_theory

        elif self.__level_of_theory_electronic != level_of_theory:
            if spycci.config.STRICT_MODE == True:
                msg = "Different electronic levels of theory used for calculating properties. Clearing properties with different electronic level of theory."
                logger.warning(msg)
                self.__clear_electronic()
                self.__level_of_theory_electronic = level_of_theory

            else:
                msg = "Different electronic levels of theory used for calculating properties. Setting level of theory to undefined."
                logger.warning(msg)
                warnings.warn(msg)
                self.__level_of_theory_electronic = "Undefined"

    def __validate_vibrational(self, engine: Engine) -> None:

        level_of_theory = self.__check_engine(engine)

        logger.debug("Validating vibrational contribution")
        logger.debug(f"current: {self.__level_of_theory_vibrational}, requested: {level_of_theory}")

        if self.__level_of_theory_vibrational is None:
            self.__level_of_theory_vibrational = level_of_theory

            if self.__pka is not None:
                if spycci.config.STRICT_MODE == True:
                    msg = "Added vibrational contribution. Clearing pKa computed with electronic energy only."
                    logger.warning(msg)
                    self.__pka = None
                else:
                    msg = (
                        "Added vibrational contribution to Properties with pKa previously computed with electronic energy only."
                    )
                    logger.warning(msg)
                    warnings.warn(msg)

        elif self.__level_of_theory_vibrational != level_of_theory:

            if spycci.config.STRICT_MODE == True:
                msg = "Different vibrational levels of theory used for calculating properties. Clearing properties with different vibrational level of theory."
                logger.warning(msg)
                self.__clear_vibrational()
                self.__level_of_theory_vibrational = level_of_theory

            else:
                msg = "Different vibrational levels of theory used for calculating properties. Setting level of theory to undefined."
                logger.warning(msg)
                warnings.warn(msg)
                self.__level_of_theory_vibrational = "Undefined"

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
        data["Helmholtz energy (Eh)"] = self.__helmholtz_free_energy
        data["pKa"] = self.__pka
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
        obj.__helmholtz_free_energy = data["Helmholtz energy (Eh)"]
        obj.__pka = data["pKa"]
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
    def helmholtz_free_energy(self) -> float:
        """
        The Helmholtz free energy of the system in Hartree.

        Returns
        -------
        float
            The Helmholtz free energy of the system in Hartree.
        """
        return self.__helmholtz_free_energy

    def set_helmholtz_free_energy(
        self,
        value: float,
        electronic_engine: Union[Engine, str],
        vibrational_engine: Union[Engine, str],
    ) -> float:
        """
        Sets the Helmholtz free energy of the system.

        Arguments
        ---------
        value: float
            The Helmholtz free energy of the system in Hartree.
        electronic_engine: Union[Engine, str]
            The engine used in the electronic calculation.
        vibrational_engine: Union[Engine, str]
            The engine used in the vibrational calculation.
        """
        logger.debug("Setting Helmholtz free energy")
        self.__validate_electronic(electronic_engine)
        self.__validate_vibrational(vibrational_engine)
        self.__helmholtz_free_energy = value

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
            return self.__electronic_energy + self.__free_energy_correction
        else:
            return None

    @property
    def pka(self) -> float:
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
        value: float,
        electronic_engine: Union[Engine, str],
        vibrational_engine: Union[Engine, str] = None,
    ) -> float:
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
        """
        logger.debug("Setting pKa")
        self.__validate_electronic(electronic_engine)
        if vibrational_engine is not None:
            self.__validate_vibrational(vibrational_engine)
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
