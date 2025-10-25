import mogli

from typing import List, Tuple, Callable, Dict, Any, Optional

from spycci.constants import atoms_dict
from spycci.systems import System


# Set default values of bond radius and gray shade
mogli.BOND_RADIUS = 0.05
mogli.BOND_GRAY_SHADE = 0.9

# Define custom color maps to assign colors to the atoms based on the value


def RdBu(data: List[float], reversed: bool = False, symmetric: bool = True) -> List[Tuple[float]]:
    """
    Simple diverging colormap going from blue (low values) to red (high values).

    Arguments
    ---------
    data: List[float]
        The list containing all the values to be represented by the colormap
    reversed: bool
        If set to true will invert the order of the colors (red low values and blue high ones)
    symmetric: bool
        If set to True (default) will adopt a symmetric color range in respect to zero. The
        values near zero will be colored in white, negative values in blue and positive ones
        in red. If set to false will set the range of the colormap based on the minimum and
        maximum values.

    Returns
    -------
    List[Tuple[float]]
        The list containing the triplets encoding the RGB coloring of each point
    """
    top, bottom = max(data), min(data)

    if symmetric:
        exc = max(top, abs(bottom))
        top, bottom = exc, -exc

    middle = (top + bottom) / 2

    colors = []
    for value in data:

        if value >= middle:
            x = (value - middle) / (top - middle)

            if reversed is False:
                colors.append((1, 1 - x, 1 - x))
            else:
                colors.append((1 - x, 1 - x, 1))

        else:
            x = (middle - value) / (middle - bottom)

            if reversed is False:
                colors.append((1 - x, 1 - x, 1))
            else:
                colors.append((1, 1 - x, 1 - x))

    return colors


def Jet(data: List[float], reversed: bool = False, clims: Optional[Tuple[float]] = None):
    """
    Simple Jet colormap.

    Arguments
    ---------
    data: List[float]
        The list containing all the values to be represented by the colormap.
    reversed: bool
        If set to true will invert the order of the colors.
    clims: Optional[Tuple[float]]
        If set to None (default) will use the minimum and maximum values of the property as
        the range of the colormap. Else, if a tuple (min, max), is given as argument, the user
        specified range will be applied in defining the coloring scheme.

    Raises
    ------
    ValueError
        Exception raised if one or more datapoints fall outside the user-defined clims.

    Returns
    -------
    List[Tuple[float]]
        The list containing the triplets encoding the RGB coloring of each point
    """

    if clims is None:
        bottom, top = min(data), max(data)
    else:
        if any([x < clims[0] for x in data]) or any([x > clims[1] for x in data]):
            raise ValueError("Data points are located outside the specified clims range.")

        bottom, top = clims

    m, c = 1 / 8, 1 / 2

    colors = []
    for value in data:

        x = 32.0 * (value - bottom) / (top - bottom)

        if reversed is True:
            x = 32 - x

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

    return colors


class MogliViewer:
    """
    Simple molecular viewer based on the `mogli` python package.

    Arguments
    ---------
    mol: System
        The `System` object containing the molecule to visualize.
    width: int
        The width of the representation in pixels.
    height: int
        The height of the representation in pixels.
    """

    def __init__(self, mol: System, width: int = 1920, height: int = 1080) -> None:
        self.__title = mol.name
        self.__width = width
        self.__height = height
        self.__mogli_mol = mogli.Molecule(mol.geometry.atomic_numbers, mol.geometry.coordinates)

    def apply_coloring(self, data: List[float], cmap: Callable = RdBu, kwargs: Dict[str, Any] = {}) -> None:
        """
        Apply a coloring to each atom in the molecule based on a list of float values.

        Arguments
        ---------
        data: List[float]
            The ordered list containing all the values to be represented by colors of the atoms.
        cmap: Callable
            The colormap funtion to be used in rendering the color of the data
        kwargs: Dict[str, Any]
            The dictionary containing the keyworded arguments to be used by the cmap function

        Raises
        ------
        ValueError
            Exception raised if the length of the data array does not match the number of atoms
            in the given molecule.
        """
        if self.__mogli_mol.atom_count != len(data):
            raise ValueError("Mismatch between given data and the number of atoms in the molecule")

        self.__mogli_mol.atom_colors = cmap(data, **kwargs)

    def show(self, camera: Optional[tuple] = None) -> None:
        """
        Opens an interactive windows where the molecule can be visualized.

        Arguments
        ---------
        camera: Optional[tuple]
            The view matrix by getting the position of the camera, the position of the center of focus and the direction
            which should point up.
        """
        mogli.show(
            self.__mogli_mol,
            width=self.__width,
            height=self.__height,
            bonds_param=1.5,
            title=self.__title,
            camera=camera,
        )

    def export(self, path: str, camera: Optional[tuple] = None) -> None:
        """
        Saves an image of the molecule.

        Arguments
        ---------
        path: str
            The path of the file to be saved.
        camera: Optional[tuple]
            The view matrix by getting the position of the camera, the position of the center of focus and the direction
            which should point up.
        """
        mogli.export(
            self.__mogli_mol,
            path,
            width=self.__width,
            height=self.__height,
            bonds_param=1.5,
            camera=camera,
        )
