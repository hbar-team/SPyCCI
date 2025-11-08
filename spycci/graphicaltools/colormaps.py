import warnings
import numpy as np

from typing import List, Tuple, Optional, Union

NAN_COLOR = (0.5, 0.5, 0.5)

def rgb_to_hex(
    rgb: Union[Tuple[float, float, float], List[Tuple[float, float, float]], np.ndarray]
) -> Union[str, List[str]]:
    """
    Convert one or more RGB triplets (each value in [0,1]) to hexadecimal color string(s) '#RRGGBB'.

    Parameters
    ----------
    rgb : tuple or list/array of tuples
        A single RGB triplet, or a list/array of RGB triplets. Each value must be in the range [0, 1].

    Returns
    -------
    str or list of str
        The HEX color string (or list of HEX strings if input is a sequence).
    """
    # Handle the case of a single color (single triplet of RGB values)
    if len(np.shape(rgb)) == 1 and len(rgb) == 3:
        r, g, b = [int(255 * np.clip(c, 0, 1)) for c in rgb]
        return f"#{r:02X}{g:02X}{b:02X}"

    # Handle the case of a list/array colors (list of RGB triplets)
    if len(np.shape(rgb)) != 2 or np.shape(rgb)[1] != 3:
        raise ValueError("Input must be a single RGB triplet or a list/array of triplets.")

    return [rgb_to_hex(c) for c in rgb]


def RdBu(data: List[float], reversed: bool = False, symmetric: bool = True) -> List[str]:
    """
    Simple diverging colormap going from blue (low values) to red (high values).

    Arguments
    ---------
    data: List[float]
        The list containing all the scalar values to be mapped to colors.
    reversed: bool
        If `True`, will invert the order of the colors (red for low values and blue for high ones)
    symmetric: bool
        If set to `True` (default) will adopt a symmetric color range with respect to zero. The
        values near zero will be colored in white, negative values in blue and positive ones
        in red (if reversed is `False`). If set to `False` will set the range of the colormap 
        based on the minimum and maximum values.

    Returns
    -------
    List[str]
        The list containing the HEX strings encoding the RGB coloring of each point
    """
    colors = []
    data = np.asarray(data, dtype=float)
    top, bottom = np.nanmax(data), np.nanmin(data)

    # If the user requested a synnetric colormap, symmetrize the colormap interval
    if symmetric:
        lim = max(abs(top), abs(bottom))
        bottom, top = -lim, lim

    # Early exit if all values are (almost) identical
    if abs(top - bottom) < 1e-12:
        warnings.warn("RdBu: all data values are (almost) identical. Uniform color has been applied.", UserWarning)
        return [(1., 1., 1.) for _ in data]

    middle = (top + bottom) / 2.
    
    for value in data:

        # Use gray to represent NaN values
        if np.isnan(value):
            colors.append(NAN_COLOR)
            continue
        
        # Conversion for the upper part of the data
        if value >= middle:
            x = (value - middle) / (top - middle)

            if reversed is False:
                colors.append((1, 1 - x, 1 - x))
            else:
                colors.append((1 - x, 1 - x, 1))

        # Conversion for the lower part of the data
        else:
            x = (middle - value) / (middle - bottom)

            if reversed is False:
                colors.append((1 - x, 1 - x, 1))
            else:
                colors.append((1, 1 - x, 1 - x))

    return colors


def Jet(data: List[float], reversed: bool = False, clims: Optional[Tuple[float, float]] = None) -> List[str]:
    """
    Simple Jet colormap.

    Arguments
    ---------
    data: List[float]
        The list containing all the scalar values to be mapped to colors.
    reversed: bool
        If set to `True` will invert the order of the colors.
    clims: Optional[Tuple[float, float]]
        If set to None (default) will use the minimum and maximum values of the property as
        the range of the colormap. Else, if a tuple (min, max), is given as argument, the user
        specified range will be applied in defining the coloring scheme.

    Raises
    ------
    ValueError
        Exception raised if one or more datapoints fall outside the user-defined clims.

    Returns
    -------
    List[str]
        The list containing the HEX strings encoding the RGB coloring of each point
    """
    colors = []
    data = np.asarray(data, dtype=float)

    # Set top and bottom values based on user settings or provided data
    if clims is None:
        bottom, top = np.min(data), np.max(data)
    else:
        bottom, top = clims
        if np.any(data < bottom) or np.any(data > top):
            raise ValueError("Data points are located outside the specified `clims` range.")
        
    # Early exit if all values are (almost) identical
    if abs(top - bottom) < 1e-12:
        warnings.warn("Jet: all data values are (almost) identical. Uniform color has been applied.", UserWarning)
        return [(1., 1., 1.) for _ in data]

    m, c = 1./8., 1./2.

    for value in data:

        # Use gray to represent NaN values
        if np.isnan(value):
            colors.append(NAN_COLOR)
            continue

        x = 32. * (value - bottom) / (top - bottom)

        if reversed is True:
            x = 32. - x

        if x <= 4:
            colors.append((0, 0, m * x + c))
        elif x <= 12:
            colors.append((0, m * (x - 4), 1))
        elif x <= 20:
            colors.append((m * (x - 12), 1, 1 - m * (x - 12)))
        elif x <= 28:
            colors.append((1, 1 - m * (x - 20), 0))
        elif x <= 32:
            colors.append((1 - m * (x - 28), 0, 0))
        else:
            # For safety: clamp overflow
            colors.append((1.0, 0.0, 0.0))

    return colors