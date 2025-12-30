from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union

def normalized_gaussian(x: Union[float, np.ndarray], c: float, FWHM: float) -> Union[float, np.ndarray]:
    """
    Normalized gaussian function

    Arguments
    ---------
    x: Union[float, np.ndarray]
        The position or the array of positions at wich the function must be evaluated.
    c: float
        The center of the distribution
    FWHM: float
        The value of the full width at half maximum of the distribution
    
    Returns
    -------
    Union[float, np.ndarray]
        The values of the function at the specified points.
    """
    sigma = FWHM/(2*np.sqrt(2*np.log(2)))
    return np.exp(-(x-c)**2/(2*sigma**2))/np.sqrt(2.*np.pi*sigma**2)

def normalized_lorentzian(x: Union[float, np.ndarray], c: float, FWHM: float) -> Union[float, np.ndarray]:
    """
    Normalized lorentzian function

    Arguments
    ---------
    x: Union[float, np.ndarray]
        The position or the array of positions at wich the function must be evaluated.
    c: float
        The center of the distribution
    FWHM: float
        The value of the full width at half maximum of the distribution
    
    Returns
    -------
    Union[float, np.ndarray]
        The values of the function at the specified points.
    """
    gamma = FWHM/2.
    return (1./np.pi) * gamma /((x - c)**2 + gamma**2)


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
        
        for i, frequency in enumerate(self.frequencies):

            intensity = None
            for j, ir_int in self.ir_transitions:
                if j == i:
                    intensity = ir_int
                    break
            
            intensity = "" if intensity is None else "{:.2f}  ".format(intensity)             
            
            info += f" {i:<6}{frequency:>11.2f}{intensity:>11}\n"

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
        obj.ir_transitions = [tuple(x) for x in data["ir_transitions"]]
        obj.ir_combination_bands = [tuple(x) for x in data["ir_combination_bands"]]
        obj.raman_transitions = [tuple(x) for x in data["raman_transitions"]]

        return obj
    
    def show_ir_spectrum(
            self,
            lineshape: Optional[str] = None,
            FWHM: float = 25.,
            range: Optional[Tuple[float, float]] = None,
            resolution: float = 0.025,
            padding: float = 200.,
            include_overtones: bool = True,
            show_bars: bool = False,
            logscale: bool = False,
            figsize: Tuple[int, int] = (12, 6),
            color: str = "#154C79",
            export_path: Optional[str] = None,
            export_dpi: int = 600,
            show: bool = True,
            axes: Optional[plt.Axes] = None,
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
        range: Optional[Tuple[float, float]]
            The interval of wavenumbers that define the reagion of the spectrum to plot. If set to `None` (default) will
            use the padding option to compute the spectrum range.
        resolution: float
            The distance (in cm^-1) between subsequent points in the spectrum plot. (default: 0.025)
        padding: float
            The padding to be used in plotting the spectrum. If set to 0, will plot the spectrum between the highest and
            lowest wavenumbers associated to the IR-active transitions (default: 200). If `range` is set, this option will
            be ignored.
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
        axes: Optional[matplotlib.pyplot.Axes]
            If given a `matplotlib.pyplot.Axes` argument, the function will add the infrared spectum to the user provided
            axes system. Beware that the `export_path` and `show` options will be automatically ignored if `axes` is set.
        
        Raises
        ------
        TypeError
            Exception raised when an invalid lineshape is given as the broadening argument.
        """
        # Define a dictionary storing the frequency of each band with the associated integrated intensity value
        bands = {}

        # Extract all the intensity values for fundamental IR transitions
        for mode, intensity in self.ir_transitions:
                
            if intensity == 0:
                continue

            frequency = self.frequencies[mode]
            if frequency not in bands:
                bands[frequency] = intensity
            else:
                bands[frequency] += intensity
        
        # If available and required by the user extract all the intensity values for the combination bands
        if self.ir_combination_bands != [] and include_overtones is True:
            for mode1, mode2, intensity in self.ir_combination_bands:
                
                if intensity == 0:
                    continue

                frequency = self.frequencies[mode1] + self.frequencies[mode2]
                if frequency not in bands:
                    bands[frequency] = intensity
                else:
                    bands[frequency] += intensity
        
        # Set the limit values of the spectrum to be plotted
        if range is None:
            fmin, fmax = min(bands.keys())-padding, max(bands.keys())+padding
        else:
            fmin, fmax = min(range), max(range)

        # Check if the user provided an `ax` argument, if not create one.
        if axes is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = axes
            fig = ax.figure

        if logscale:
            ax.set_yscale("log")

        if lineshape is not None:
            lineshape = lineshape.lower()

        # If the lineshape is set to None just present a stem plot with the integrated intensities values
        if lineshape is None:
            ax.stem(bands.keys(), bands.values(), linefmt=color, basefmt="None", markerfmt="None")
            ax.set_xlim((fmin, fmax))

            if logscale is False:
                ax.set_ylim(bottom=0)
        
        # If the user requested a lineshape compute the spectrum by summing the contribution of each band
        # Compute each contribution by vectorizing over the frequency range. Each contribution is expressed
        # as the product of a normalized lineshape function by the integrated intensity of the band.
        elif lineshape in ["lorentzian", "gaussian"]:
            
            frequencies = np.arange(fmin, fmax, resolution)
            total_intensity = np.zeros_like(frequencies)
            
            # Compute each band contribution to the total intensity
            for f0, intensity in bands.items():

                if lineshape == "lorentzian":
                    total_intensity += intensity*normalized_lorentzian(frequencies, f0, FWHM)

                elif lineshape == "gaussian":
                    total_intensity += intensity*normalized_gaussian(frequencies, f0, FWHM)

            # Plot the obtained intensity values
            ax.plot(frequencies, total_intensity, color=color, linewidth=1.5)
            ax.set_xlim((fmin, fmax))

            if show_bars:
                ax2 = ax.twinx()
                ax2.stem(bands.keys(), bands.values(), linefmt=color, basefmt="None", markerfmt="None")
                ax2.tick_params(axis="y", labelsize=16)
                ax2.set_ylabel(r"Integrated Intensity [$km/mol$]", fontsize=20)
        
        else:
            raise TypeError(f"`{lineshape}` lineshape option is invalid.")
        
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xlabel(r"Wavenumber [$cm^{-1}$]", fontsize=20)
        ax.set_ylabel(r"Intensity [$km/mol$]", fontsize=20)
        
        ax.grid(which="major", color="#DDDDDD")
        ax.grid(which="minor", color="#EEEEEE")

        if axes is None:
            
            plt.tight_layout()

            if export_path is not None:
                plt.savefig(export_path, dpi=export_dpi)

            if show:
                plt.show()

    
    def show_raman_spectrum(
            self,
            lineshape: Optional[str] = None,
            FWHM: float = 25.,
            range: Optional[Tuple[float, float]] = None,
            resolution: float = 0.025,
            padding: float = 200.,
            show_bars: bool = False,
            logscale: bool = False,
            figsize: Tuple[int, int] = (12, 6),
            color: str = "#154C79",
            export_path: Optional[str] = None,
            export_dpi: int = 600,
            show: bool = True,
            axes: Optional[plt.Axes] = None,
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
        range: Optional[Tuple[float, float]]
            The interval of wavenumbers that define the reagion of the spectrum to plot. If set to `None` (default) will
            use the padding option to compute the spectrum range.
        resolution: float
            The distance (in cm^-1) between subsequent points in the spectrum plot. (default: 0.025)
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
        axes: Optional[matplotlib.pyplot.Axes]
            If given a `matplotlib.pyplot.Axes` argument, the function will add the raman spectum to the user provided
            axes system. Beware that the `export_path` and `show` options will be automatically ignored if `axes` is set.
        
        Raises
        ------
        TypeError
            Exception raised when an invalid lineshape is given as the broadening argument.
        """
        # Define a dictionary storing the frequency of each band with the associated acitivty value
        bands = {}

        # Extract all the activity values for each Raman transition
        for mode, activity, _ in self.raman_transitions:
                
            if activity == 0:
                continue

            frequency = self.frequencies[mode]
            if frequency not in bands:
                bands[frequency] = activity
            else:
                bands[frequency] += activity

        # Set the limit values of the spectrum to be plotted
        if range is None:
            fmin, fmax = min(bands.keys())-padding, max(bands.keys())+padding
        else:
            fmin, fmax = min(range), max(range)

        # Check if the user provided an `ax` argument, if not create one.
        if axes is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = axes
            fig = ax.figure

        if logscale:
            ax.set_yscale("log")

        if lineshape is not None:
            lineshape = lineshape.lower()

        # If the lineshape is set to None just present a stem plot with the activity values
        if lineshape is None:
            ax.stem(bands.keys(), bands.values(), linefmt=color, basefmt="None", markerfmt="None")
            ax.set_xlim((fmin, fmax))

            if logscale is False:
                ax.set_ylim(bottom=0)
        
        # If the user requested a lineshape compute the spectrum by summing the contribution of each band
        # Compute each contribution by vectorizing over the frequency range. Each contribution is expressed
        # as the product of a normalized lineshape function by the activity of the band.
        elif lineshape in ["lorentzian", "gaussian"]:
            
            frequencies = np.arange(fmin, fmax, resolution)
            total_intensity = np.zeros_like(frequencies)

            for f0, intensity in bands.items():
                
                if lineshape == "lorentzian":
                    total_intensity += intensity*normalized_lorentzian(frequencies, f0, FWHM)

                elif lineshape == "gaussian":
                    total_intensity += intensity*normalized_gaussian(frequencies, f0, FWHM)
            
            # Plot the obtained activity values
            ax.plot(frequencies, total_intensity, color=color, linewidth=1.5)

            if show_bars:
                ax.stem(bands.keys(), bands.values(), linefmt=color, basefmt="None", markerfmt="None")
        
        else:
            raise TypeError(f"`{lineshape}` lineshape option is invalid.")
        
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xlabel(r"Wavenumber [$cm^{-1}$]", fontsize=20)
        ax.set_ylabel(r"Activity", fontsize=20)
        
        ax.grid(which="major", color="#DDDDDD")
        ax.grid(which="minor", color="#EEEEEE")

        if axes is None:
            
            plt.tight_layout()

            if export_path is not None:
                plt.savefig(export_path, dpi=export_dpi)

            if show:
                plt.show()

    