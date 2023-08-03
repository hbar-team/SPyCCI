from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def unitary_height_gaussian(x: float, c: float, FWHM: float) -> float:
    """
    Unitary height gaussian function

    Arguments
    ---------
    x: float
        The position at wich the function must be evaluated.
    c: float
        The center of the distribution
    FWHM: float
        The value of the full width at half maximum of the distribution
    
    Returns
    -------
    float
        The value of the function at the specified point.
    """
    sigma = FWHM/(2*np.sqrt(2*np.log(2)))
    return np.exp(-(x-c)**2/(2*sigma**2))

def unitary_height_lorentzian(x, c, FWHM):
    """
    Unitary height lorentzian function

    Arguments
    ---------
    x: float
        The position at wich the function must be evaluated.
    c: float
        The center of the distribution
    FWHM: float
        The value of the full width at half maximum of the distribution
    
    Returns
    -------
    float
        The value of the function at the specified point.
    """
    gamma = FWHM/2
    return (gamma/2)**2 /((x - c)**2 + (gamma/2)**2)


class VibrationalData:
    """
    The `VibrationalData` class holds all the information related to the vibrational properties of a given system.

    Attributes
    ----------
    frequencies: List[float]
        The list of frequencies associated with each mode of vibration. The leading modes, associated to rotations and
        translations will be associated with zero frequencies.
    normal_modes: List[np.ndarray]
        The vectors encoding the cartesian displacements associated to each mode of vibration.
    ir_transitions: List[Tuple[int, float]]
        The list of tuples associated with each infrared transition. Each tuple is composed by the index of the mode
        involved and the corresponding intensity in km/mol.
    ir_combination_bands: List[Tuple[int, int, float]]
        The list of tuples associated with each combination and overtone transition. Each tuple is composed by the
        index of the two modes involved and the corresponding intensity in km/mol.
    raman_transitions: List[Tuple[int, float, float]]
        The list of tuples associated with each Raman transition. Each tuple is composed by the index of the mode
        involved, the corresponding activity and depolarization.
    """
    def __init__(self) -> None:
        self.frequencies: List[float] = []
        self.normal_modes: List[np.ndarray] = []
        self.ir_transitions: List[Tuple[int, float]] = []
        self.ir_combination_bands: List[Tuple[int, int, float]] = []
        self.raman_transitions: List[Tuple[int, float, float]] = []
    
    def __str__(self) -> str:
        info = "VIBRATIONAL FREQUENCIES\n"
        info += "----------------------------------------------\n"
        info += " index  frequency  intensity \n"
        info += "         (cm^-1)   (km/mol)  \n"
        info += "----------------------------------------------\n"
        for mode, frequency in enumerate(self.frequencies):

            ir_intensity = None
            for m, intensity in self.ir_transitions:
                if m == mode:
                    ir_intensity = intensity
                    break
            
            if ir_intensity is None:
                ir_intensity = ""
            else:
                ir_intensity = "{:.2f}  ".format(ir_intensity)   

            frequency = "{:.2f}  ".format(frequency)            
            
            info += f" {mode:<6}{frequency:>11}{ir_intensity:>11}\n"

        info += "\n"

        return info

        
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
        data["frequencies"] = self.frequencies
        data["normal_modes"] = [list(x) for x in self.normal_modes]
        data["ir_transitions"] = self.ir_transitions
        data["ir_combination_bands"] = self.ir_combination_bands
        data["raman_transitions"] = self.raman_transitions
        return data

    @classmethod
    def from_dict(cls, data: dict) -> VibrationalData:
        """
        Construct a VibrationalData object from the data encoded in a dictionary.

        Arguments
        ---------
        data: dict
            The dictionary containing the class attributes

        Returns
        -------
        Properties
            The fully initialized VibrationalData object
        """
        obj = cls()
        obj.frequencies = data["frequencies"]
        obj.normal_modes = [np.array(x) for x in data["normal_modes"]]
        obj.ir_transitions = [(x, y) for x, y in data["ir_transitions"]]
        obj.ir_combination_bands = [(x, y, z) for x, y, z in data["ir_combination_bands"]]
        obj.raman_transitions = data["raman_transitions"]
        return obj
    
    def show_ir_spectrum(
            self,
            lineshape: Optional[str] = None,
            FWHM: float = 25.,
            padding: float = 200.,
            include_overtones: bool = True,
            show_bars: bool = False,
            logscale: bool = False,
            figsize: Tuple[int, int] = (12, 6),
            color: str = "#154C79",
            export_path: Optional[str] = None,
            export_dpi: int = 600,
            show: bool = True
        ) -> None:
        """
        Plots the infrared spectrum of the molecule.

        Arguments
        ---------
        lineshape: Optional[str]
            The type of broadening to be used in rendering the spectrum. The available lineshapes are `lorentzian` and
            `gaussian`. If set to `None` only vertical bars will be used to represent the spectrum.
        FWHM: float
            The full width at half maximum in cm^-1 of the broadening lineshapes (default: 25).
        padding: float
            The padding to be used in plotting the spectrum. If set to 0, will plot the spectrum between the highest and
            lowest wavenumbers associated to the IR-active transitions (default: 200).
        include_overtones: bool
            If set to True (default) will use, if available, the overtones and combination bands to plot the spectrum.
        show_bars: bool
            If set to True will show the intensity bars even if the lineshape parameters is set.
        logscale: bool
            If set to True will use a logaritmic scale to represent the intensities (default: False). The option is mainly
            useful when plotting the spectrum without broadening.
        figsize: Tuple[int, int]
            The size of the matplotlib figure.
        color: str
            The string encoding the color of the line.
        export_path: Optional[str]
            The string encoding the location in which a copy of the spectrum should be saved. If set to None (default)
            no file will be saved.
        export_dpi: int
            The resolution of the exported image (default: 600).
        show: bool
            If set to True (default) will open an interactive window containing the spectrum.
        
        Raises
        ------
        TypeError
            Exception raised when an invalid lineshape is given as the broadening argument.
        """

        ir_bands = {}
        for mode, intensity in self.ir_transitions:
                
            if intensity == 0:
                continue

            frequency = self.frequencies[mode]
            if frequency not in ir_bands:
                ir_bands[frequency] = intensity
            else:
                ir_bands[frequency] += intensity
        

        if self.ir_combination_bands != [] and include_overtones is True:
            for mode1, mode2, intensity in self.ir_combination_bands:
                
                if intensity == 0:
                    continue

                frequency = self.frequencies[mode1] + self.frequencies[mode2]
                if frequency not in ir_bands:
                    ir_bands[frequency] = intensity
                else:
                    ir_bands[frequency] += intensity

        fmin, fmax = min(ir_bands.keys())-padding, max(ir_bands.keys())+padding

        fig = plt.figure(figsize=figsize)

        if logscale:
            plt.yscale("log")

        if lineshape is None:
            plt.stem(ir_bands.keys(), ir_bands.values(), linefmt=color, basefmt="None", markerfmt="None")
            plt.xlim((fmin, fmax))

            if logscale is False:
                plt.ylim(bottom=0)
        
        else:
            amplitude = []
            frequency = np.arange(fmin, fmax, 0.01)
            for f in frequency:
                value = 0
                for f0, intensity in ir_bands.items():
                    
                    if lineshape == "lorentzian":
                        value += intensity*unitary_height_lorentzian(f, f0, FWHM)

                    elif lineshape == "gaussian":
                        value += intensity*unitary_height_gaussian(f, f0, FWHM)
                    
                    else:
                        raise TypeError(f"`{lineshape}` lineshape option is invalid.")
                
                amplitude.append(value)

            plt.plot(frequency, amplitude, color=color, linewidth=1.5)

            if show_bars:
                plt.stem(ir_bands.keys(), ir_bands.values(), linefmt=color, basefmt="None", markerfmt="None")
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r"Wavenumber [$cm^{-1}$]", fontsize=20)
        plt.ylabel(r"Intensity [$km/mol$]", fontsize=20)
        
        plt.grid(which="major", color="#DDDDDD")
        plt.grid(which="minor", color="#EEEEEE")

        plt.tight_layout()

        if export_path is not None:
            plt.savefig(export_path, dpi=export_dpi)

        if show:
            plt.show()

    
    def show_raman_spectrum(
            self,
            lineshape: Optional[str] = None,
            FWHM: float = 25.,
            padding: float = 200.,
            show_bars: bool = False,
            logscale: bool = False,
            figsize: Tuple[int, int] = (12, 6),
            color: str = "#154C79",
            export_path: Optional[str] = None,
            export_dpi: int = 600,
            show: bool = True
        ) -> None:
        """
        Plots the raman spectrum of the molecule.

        Arguments
        ---------
        lineshape: Optional[str]
            The type of broadening to be used in rendering the spectrum. The available lineshapes are `lorentzian` and
            `gaussian`. If set to `None` only vertical bars will be used to represent the spectrum.
        FWHM: float
            The full width at half maximum in cm^-1 of the broadening lineshapes (default: 25).
        padding: float
            The padding to be used in plotting the spectrum. If set to 0, will plot the spectrum between the highest and
            lowest wavenumbers associated to the IR-active transitions (default: 200).
        show_bars: bool
            If set to True will show the intensity bars even if the lineshape parameters is set.
        logscale: bool
            If set to True will use a logaritmic scale to represent the intensities (default: False). The option is mainly
            useful when plotting the spectrum without broadening.
        figsize: Tuple[int, int]
            The size of the matplotlib figure.
        color: str
            The string encoding the color of the line.
        export_path: Optional[str]
            The string encoding the location in which a copy of the spectrum should be saved. If set to None (default)
            no file will be saved.
        export_dpi: int
            The resolution of the exported image (default: 600).
        show: bool
            If set to True (default) will open an interactive window containing the spectrum.
        
        Raises
        ------
        TypeError
            Exception raised when an invalid lineshape is given as the broadening argument.
        """
        raman_bands = {}
        for mode, activity, _ in self.raman_transitions:
                
            if activity == 0:
                continue

            frequency = self.frequencies[mode]
            if frequency not in raman_bands:
                raman_bands[frequency] = activity
            else:
                raman_bands[frequency] += activity

        fmin, fmax = min(raman_bands.keys())-padding, max(raman_bands.keys())+padding

        fig = plt.figure(figsize=figsize)

        if logscale:
            plt.yscale("log")

        if lineshape is None:
            plt.stem(raman_bands.keys(), raman_bands.values(), linefmt=color, basefmt="None", markerfmt="None")
            plt.xlim((fmin, fmax))

            if logscale is False:
                plt.ylim(bottom=0)
        
        else:
            amplitude = []
            frequency = np.arange(fmin, fmax, 0.01)
            for f in frequency:
                value = 0
                for f0, intensity in raman_bands.items():
                    
                    if lineshape == "lorentzian":
                        value += intensity*unitary_height_lorentzian(f, f0, FWHM)

                    elif lineshape == "gaussian":
                        value += intensity*unitary_height_gaussian(f, f0, FWHM)
                    
                    else:
                        raise TypeError(f"`{lineshape}` lineshape option is invalid.")
                
                amplitude.append(value)

            plt.plot(frequency, amplitude, color=color, linewidth=1.5)

            if show_bars:
                plt.stem(raman_bands.keys(), raman_bands.values(), linefmt=color, basefmt="None", markerfmt="None")
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r"Wavenumber [$cm^{-1}$]", fontsize=20)
        plt.ylabel(r"Activity", fontsize=20)
        
        plt.grid(which="major", color="#DDDDDD")
        plt.grid(which="minor", color="#EEEEEE")

        plt.tight_layout()

        if export_path is not None:
            plt.savefig(export_path, dpi=export_dpi)

        if show:
            plt.show()

    
