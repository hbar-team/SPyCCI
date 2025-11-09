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

# ---------------------------------------------------------------------------------------------------------
#                                     DEFINITION OF GENERIC COLORMAPS                                      
# ---------------------------------------------------------------------------------------------------------

def DivergingColormap(
        data: List[float],
        ctop: Tuple[float, float, float],
        cbottom: Tuple[float, float, float],
        cmiddle: Tuple[float, float, float] = (1., 1., 1.),
        reversed: bool = False,
        symmetric: bool = True
    ) -> List[Tuple[float, float, float]]:
    """
    Simple diverging colormap going from a top color (`ctop`) to a bottom one (`cbottom`) with 
    white in the middle.

    Arguments
    ---------
    data: List[float]
        The list containing all the scalar values to be mapped to colors.
    ctop: Tuple[float, float, float],
        The triplet of RGB values encoding the top color.
    cbottom: Tuple[float, float, float],
        The triplet of RGB values encoding the bottom color.
    cmiddle: Tuple[float, float, float],
        The triplet of RGB values encoding the middle color. (default: (1., 1., 1.))
    reversed: bool
        If `True`, will invert the order of the colors (red for low values and blue for high ones)
    symmetric: bool
        If set to `True` (default) will adopt a symmetric color range with respect to zero. The
        values near zero will be colored in white, negative values in blue and positive ones
        in red (if reversed is `False`). If set to `False` will set the range of the colormap 
        based on the minimum and maximum values.

    Returns
    -------
    List[Tuple[float, float, float]]
        The list containing the RGB triplets (as tuples of float) encoding the RGB coloring of each point
    """
    colors = []
    data = np.asarray(data, dtype=float)
    top, bottom = np.nanmax(data), np.nanmin(data)

    # Check the user provided colors and convert them to np.array and clip them for safety
    if len(ctop) != 3 or len(cbottom) != 3:
        raise ValueError("DivergingColormap: The top and bottom colors must be triplets of RGB values")
    
    ctop = np.clip(np.asarray(ctop, dtype=float), 0.0, 1.0)
    cbottom = np.clip(np.asarray(cbottom, dtype=float), 0.0, 1.0)
    cmiddle = np.clip(np.asarray(cmiddle, dtype=float), 0.0, 1.0)

    # Set upper and lower colors based on `reversed` state
    cupper = ctop if reversed is False else cbottom
    clower = cbottom if reversed is False else ctop

    # If the user requested a synnetric colormap, symmetrize the colormap interval
    if symmetric:
        lim = max(abs(top), abs(bottom))
        bottom, top = -lim, lim

    # Early exit if all values are (almost) identical
    if abs(top - bottom) < 1e-12:
        warnings.warn("DivergingColormap: all data values are (almost) identical. Uniform color has been applied.", UserWarning)
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
            color = (1.-x) * cmiddle + x * cupper
            color = tuple(np.clip(color, 0., 1.))
            colors.append(color)

        # Conversion for the lower part of the data
        else:
            x = (middle - value) / (middle - bottom)
            color = (1.-x) * cmiddle + x * clower
            color = tuple(np.clip(color, 0., 1.))
            colors.append(color)

    return colors


def PolynomialColormap(
        data: List[float],
        coefficients: Union[np.ndarray, List[List[float]]],
        reversed: bool = False,
        clims: Optional[Tuple[float, float]] = None
    ) -> List[Tuple[float, float, float]]:
    """
    Generic definition of a polynomial colormap. The intensity (`I_i`) of the color for each channel
    is defined as: `I_i = c_i[0] + c_i[1]*x + c_i[2]*x^2 + ... + c_i[n]*x^n`, where `c_i` represents
    the list of coefficents for the `i`-th channel.

    Arguments
    ---------
    data: List[float]
        The list containing all the scalar values to be mapped to colors.
    coefficients: Union[np.ndarray, List[List[float]]]
        The 3xN matrix holding, by row (first idex), the polynomial coefficents `c_i` for the 3 RGB channels.
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
    List[Tuple[float, float, float]]
        The list containing the RGB triplets (as tuples of float) encoding the RGB coloring of each point
    """
    colors = []
    coeff = np.asarray(coefficients)

    # Check the user provided list of RGB polynomial coefficients
    if len(coeff.shape) !=2 or coeff.shape[0] != 3:
        raise ValueError("PolynomialColormap: The coefficients matrix must be a 3xN matrix.")

    # Set top and bottom values based on user settings or provided data
    if clims is None:
        bottom, top = np.min(data), np.max(data)
    else:
        bottom, top = clims
        if np.any(data < bottom) or np.any(data > top):
            raise ValueError("Data points are located outside the specified `clims` range.")
    
    for value in data:

        # Use gray to represent NaN values
        if np.isnan(value):
            colors.append(NAN_COLOR)
            continue
        
        # Normalize x in [0,1]
        x = (value - bottom)/(top - bottom)
        if reversed:
            x = 1. - x

        # Evaluate polynomial for each channel
        rgb = [np.polyval(coeff[i][::-1], x) for i in range(3)]
        
        # Clamp to [0, 1]
        rgb = tuple(np.clip(rgb, 0.0, 1.0))
        colors.append(rgb)
    
    return colors

# ---------------------------------------------------------------------------------------------------------
#                                     DEFINITION OF DERIVED COLORMAPS                                      
# ---------------------------------------------------------------------------------------------------------

def RdBu(data: List[float], reversed: bool = False, symmetric: bool = True) -> List[Tuple[float, float, float]]:
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
    List[Tuple[float, float, float]]
        The list containing the RGB triplets (as tuples of float) encoding the RGB coloring of each point
    """
    cmap = DivergingColormap(
        data=data,
        ctop=(1., 0., 0.),
        cbottom=(0., 0., 1.),
        reversed=reversed,
        symmetric=symmetric,
    )

    return cmap


def PiYG(data: List[float], reversed: bool = False, symmetric: bool = True) -> List[Tuple[float, float, float]]:
    """
    Simple diverging colormap going from green (low values) to pink (high values).

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
    List[Tuple[float, float, float]]
        The list containing the RGB triplets (as tuples of float) encoding the RGB coloring of each point
    """
    cmap = DivergingColormap(
        data=data,
        ctop=(0.890, 0.008, 0.969),
        cbottom=(0.008, 0.969, 0.137),
        reversed=reversed,
        symmetric=symmetric,
    )

    return cmap


def RdYlBu(data: List[float], reversed: bool = False, symmetric: bool = True) -> List[Tuple[float, float, float]]:
    """
    Simple diverging colormap going from blue (low values) to red (high values) using yellow as the central color.

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
    List[Tuple[float, float, float]]
        The list containing the RGB triplets (as tuples of float) encoding the RGB coloring of each point
    """
    cmap = DivergingColormap(
        data=data,
        ctop=(1., 0., 0.),
        cbottom=(0., 0., 1.),
        cmiddle=(1.0, 0.933, 0.443),
        reversed=reversed,
        symmetric=symmetric,
    )

    return cmap


def Jet(data: List[float], reversed: bool = False, clims: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float, float]]:
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
    List[Tuple[float, float, float]]
        The list containing the RGB triplets (as tuples of float) encoding the RGB coloring of each point
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

    
def Turbo(data: List[float], reversed: bool = False, clims: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float, float]]:
    """
    Simple Turbo colormap.

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
    List[Tuple[float, float, float]]
        The list containing the RGB triplets (as tuples of float) encoding the RGB coloring of each point
    """

    coefficients = np.array([
        [0.13572138, 4.59736372, -42.32768975, 130.58871182, -150.56663492, 58.13745345],
        [0.09140261, 2.18561734, 4.80520480, -14.01945096, 4.21085636, 2.77473115],
        [0.10667330, 12.59256348, -60.10967552, 109.07449945, -88.50658251, 26.81826097]
    ])

    cmap = PolynomialColormap(
        data,
        coefficients=coefficients,
        reversed=reversed,
        clims=clims
        )

    return cmap


def Viridis(data: List[float], reversed: bool = False, clims: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float, float]]:
    """
    Simple Viridis colormap.

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
    List[Tuple[float, float, float]]
        The list containing the RGB triplets (as tuples of float) encoding the RGB coloring of each point
    """

    coefficients = np.array([
        [0.2777273272, 0.1050930431, -0.3308618287, -4.634230499, 6.228269936, 4.776384998, -5.435455856],
        [0.005407344545, 1.40461353, 0.2148475595, -5.799100973, 14.17993337, -13.74514538, 4.645852612],
        [0.3340998053, 1.384590163, 0.09509516303, -19.33244096, 56.6905526, -65.35303263, 26.31243525]
    ])

    cmap = PolynomialColormap(
        data,
        coefficients=coefficients,
        reversed=reversed,
        clims=clims
        )

    return cmap


def Plasma(data: List[float], reversed: bool = False, clims: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float, float]]:
    """
    Simple Plasma colormap.

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
    List[Tuple[float, float, float]]
        The list containing the RGB triplets (as tuples of float) encoding the RGB coloring of each point
    """

    coefficients = np.array([
        [0.05873234392,2.176514634,-2.689460476,6.130348346,-11.10743619,10.02306558,-3.658713843],
        [0.02643670893,0.2383834171,-7.455851136,42.34618815,-82.66631109,71.4136177,-22.93153465],
        [0.544,0.75396046,3.11079994,-28.51885465,60.13984767,-54.07218656,18.19190779]
    ])

    cmap = PolynomialColormap(
        data,
        coefficients=coefficients,
        reversed=reversed,
        clims=clims
        )

    return cmap