import mogli

from typing import List, Tuple, Callable, Dict, Any, Optional

from spycci.constants import atoms_dict
from spycci.systems import System


# Set default values of bond radius and gray shade
mogli.BOND_RADIUS = 0.05
mogli.BOND_GRAY_SHADE = 0.9


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
