import numpy as np
import pyvista as pv

from typing import Union, Optional, List, Tuple

from spycci.core.geometry import MolecularGeometry
from spycci.systems import System

# Covalent radius (in Å) for single bonds from "Pekka Pyykkö and Michiko Atsumi. Molecular Single-Bond Covalent Radii for Elements 1-118. Chemistry - A European Journal, 15(1):186–197, jan 2009"
COVALENT_RADII = {'H': 0.32, 'He': 0.46, 'Li': 1.33, 'Be': 1.02, 'B': 0.85, 'C': 0.75, 'N': 0.71, 'O': 0.63, 'F': 0.64, 'Ne': 0.67, 'Na': 1.55, 'Mg': 1.39, 'Al': 1.26, 'Si': 1.16, 'P': 1.11, 'S': 1.03, 'Cl': 0.99, 'Ar': 0.96, 'K': 1.96, 'Ca': 1.71, 'Sc': 1.48, 'Ti': 1.36, 'V': 1.34, 'Cr': 1.22, 'Mn': 1.19, 'Fe': 1.16, 'Co': 1.11, 'Ni': 1.1, 'Cu': 1.12, 'Zn': 1.18, 'Ga': 1.24, 'Ge': 1.21, 'As': 1.21, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.17, 'Rb': 2.1, 'Sr': 1.85, 'Y': 1.63, 'Zr': 1.54, 'Nb': 1.47, 'Mo': 1.38, 'Tc': 1.28, 'Ru': 1.25, 'Rh': 1.25, 'Pd': 1.2, 'Ag': 1.28, 'Cd': 1.36, 'In': 1.42, 'Sn': 1.4, 'Sb': 1.4, 'Te': 1.36, 'I': 1.33, 'Xe': 1.31, 'Cs': 2.32, 'Ba': 1.96, 'La': 1.8, 'Ce': 1.63, 'Pr': 1.76, 'Nd': 1.74, 'Pm': 1.73, 'Sm': 1.72, 'Eu': 1.68, 'Gd': 1.69, 'Tb': 1.68, 'Dy': 1.67, 'Ho': 1.66, 'Er': 1.65, 'Tm': 1.64, 'Yb': 1.7, 'Lu': 1.62, 'Hf': 1.52, 'Ta': 1.46, 'W': 1.37, 'Re': 1.31, 'Os': 1.29, 'Ir': 1.22, 'Pt': 1.23, 'Au': 1.24, 'Hg': 1.33, 'Tl': 1.44, 'Pb': 1.44, 'Bi': 1.51, 'Po': 1.45, 'At': 1.47, 'Rn': 1.42, 'Fr': 2.23, 'Ra': 2.01, 'Ac': 1.86, 'Th': 1.75, 'Pa': 1.69, 'U': 1.7, 'Np': 1.71, 'Pu': 1.72, 'Am': 1.66, 'Cm': 1.66, 'Bk': 1.68, 'Cf': 1.68, 'Es': 1.65, 'Fm': 1.67, 'Md': 1.73, 'No': 1.76, 'Lr': 1.61, 'Rf': 1.57, 'Db': 1.49, 'Sg': 1.43, 'Bh': 1.41, 'Hs': 1.34, 'Mt': 1.29, 'Ds': 1.28, 'Rg': 1.21, 'Cn': 1.22, 'Nh': 1.36, 'Fl': 1.43, 'Mc': 1.62, 'Lv': 1.75, 'Ts': 1.65, 'Og': 1.57}

# Atom colors derived from the JMOL standard (carbons have been set to darker grey value)
ATOM_COLORS = {"H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5", "C": "#282828", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00", "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7", "Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F", "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929", "Kr": "#5CB8D1", "Rb": "#702EB0", "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0", "Nb": "#73C2C9", "Mo": "#54B5B5", "Tc": "#3B9E9E", "Ru": "#248F8F", "Rh": "#0A7D8C", "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573", "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00", "I": "#940094", "Xe": "#429EB0", "Cs": "#57178F", "Ba": "#00C900", "La": "#70D4FF", "Ce": "#FFFFC7", "Pr": "#D9FFC7", "Nd": "#C7FFC7", "Pm": "#A3FFC7", "Sm": "#8FFFC7", "Eu": "#61FFC7", "Gd": "#45FFC7", "Tb": "#30FFC7", "Dy": "#1FFFC7", "Ho": "#00FF9C", "Er": "#00E675", "Tm": "#00D452", "Yb": "#00BF38", "Lu": "#00AB24", "Hf": "#4DC2FF", "Ta": "#4DA6FF", "W": "#2194D6", "Re": "#267DAB", "Os": "#266696", "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123", "Hg": "#B8B8D0", "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5", "Po": "#AB5C00", "At": "#754F45", "Rn": "#428296", "Fr": "#420066", "Ra": "#007D00", "Ac": "#70ABFA", "Th": "#00BAFF", "Pa": "#00A1FF", "U": "#008FFF", "Np": "#0080FF", "Pu": "#006BFF", "Am": "#545CF2", "Cm": "#785CE3", "Bk": "#8A4FE3", "Cf": "#A136D4", "Es": "#B31FD4", "Fm": "#B31FBA", "Md": "#B30DA6", "No": "#BD0D87", "Lr": "#C70066", "Rf": "#CC0059", "Db": "#D1004F", "Sg": "#D90045", "Bh": "#E00038", "Hs": "#E6002E", "Mt": "#EB0026", "Ds": "#EB0026", "Rg": "#E6002E", "Cn": "#E00038", "Nh": "#D90045", "Fl": "#D1004F", "Mc": "#CC0059", "Lv": "#C70066", "Ts": "#BD0D87", "Og": "#B30DA6"}


def show_molecule(
        molecule: System,
        atoms_colors: Optional[Union[List[str], List[Tuple[float, float, float]]]] = None,
        atom_scale : float = 0.4,
        bond_radius : float = 0.075,
        window_size: Tuple[int, int] = (1200, 1000),
        background : str = "#FFFFFF",
        only_single_bonds: bool = False,
        title : str = "",
        title_color: str = "#000000",
        title_size: int = 18,
        show_axes: bool = True,
        azimuth : float = 0.,
        elevation: float = 0.,
        zoom: float = 1.,
        export_path : Optional[str] = None,
        transparent_background: bool = False,
    ) -> None:
    """
    Display a 3D ball-and-stick representation of a molecular structure using PyVista. This function
    creates an interactive three-dimensional visualization of a molecule, where atoms are rendered as
    spheres and chemical bonds as cylinders. It supports single, double, triple, and aromatic bonds 
    (bond order = 1.5), with bond topology and geometry automatically inferred from a `System` object.

    Parameters
    ----------
    molecule : System
        The molecular system to visualize.
    atoms_colors : Optional[Union[List[str], List[Tuple[float, float, float]]]]
        The ordered list of HEX color values or RGB triplets to be used to color the atoms in the structure.
        If `None` (default), will use the standard JMOL coloring scheme to represent the molecule.
    atom_scale : float
        Scaling factor (default=0.4) applied to covalent radii when drawing atomic spheres. Increasing
        this value enlarges the atoms relative to bond lengths.
    bond_radius : float
        Radius of the cylindrical meshes used to represent bonds. (default: 0.075)
    window_size: Tuple[int, int]
        The tuple of two integer values encoding the window size. (default: (1200, 1000))
    background : str
        Background color of the 3D scene in hexadecimal RGB format. (default: "#222222")
    only_single_bonds : bool
        If `True`, all bonds with a nonzero bond order are displayed as single bonds, ignoring multiple
        bond representations. Useful for simplified or schematic visualizations. (default: False)
    title : str
        Optional title text displayed in the 3D scene. (default: "")
    title_color : str
        Title text color, specified as a hexadecimal RGB string. (default: "#FFFFFF")
    title_size : int 
        Font size for the title text. (default: 18)
    show_axes : bool
        If `True` (default) will show the axis orientation in the bottom-left part of the window.
    azimuth : float
        The rotation angle of the camera around the vertical axis, in degrees.Positive values rotate
        the view counterclockwise when looking from above. (default: 0.)
    elevation : float
        The vertical tilt angle of the camera, in degrees. Positive values tilt the view downward (as if
        moving the camera upward). (default: 0.)
    zoom : float
        Zoom factor applied to the camera view. Values greater than 1 zoom in (move closer), values smaller
        than 1 zoom out (move farther away). (default: 1.)
    export_path: Optional[str]
        If not `None`, will run the plotter in off-screen render mode and save a screenshot of the current view
        at the user specified path. If `None` (default) will run the viewer in interactive mode.
    transparent_background: bool
        If set to `True` will export the image with transparent background. (default: `False`)

    Notes
    -----
    - Atoms are rendered as spheres (`pv.Sphere`), with radii derived from the covalent radii scaled
      by `atom_scale`.
    - Bonds are rendered as cylinders (`pv.Cylinder`) positioned and oriented along the vector connecting
      two bonded atoms. Bond orders are represented as follows:

        * **1.0 (single bond)** → one central cylinder  
        * **2.0 (double bond)** → two parallel cylinders  
        * **3.0 (triple bond)** → three parallel cylinders  
        * **1.5 (aromatic bond)** → one solid cylinder plus one dashed line made of segments  

      For multiple bonds, a perpendicular vector is computed using a cross product
      (`np.cross`) to laterally offset the cylinders so that they do not overlap.

    - Very short interatomic distances (`< 1e-6`) are ignored to prevent
      degenerate or numerically unstable bond geometries.

    - The visualization uses smooth shading and balanced mesh resolution
      for both high-quality rendering and interactive performance.

    Raises
    ------
    TypeError
        If `molecule` is not an instance of `MolecularGeometry` or `System`.
    """
    if isinstance(molecule, System) is False:
        raise TypeError("The `molecule` arguent must be of type `System`.")
    
    plotter = pv.Plotter(
        window_size = window_size,
        off_screen = True if export_path is not None else False
    )
    plotter.set_background(background)
       
    # Obtain connectivity and extract geometry
    bond_type_matrix = molecule.bond_type_matrix
    geometry = molecule.geometry

    # Validate `atoms_color` input
    if atoms_colors is not None and len(atoms_colors) != geometry.atomcount:
        raise ValueError("The list of atoms colors must have the same length as the list of atoms in the structure.")

    # Draw atoms as spheres
    for idx, (atom, coordinates) in enumerate(zip(geometry.atoms, geometry.coordinates)):
        if atoms_colors is None:
            color = ATOM_COLORS.get(atom, '#888888')
        else:
            color = atoms_colors[idx]
        radius = COVALENT_RADII.get(atom, 0.7) * atom_scale
        sphere = pv.Sphere(radius=radius, center=coordinates, theta_resolution=24, phi_resolution=24)
        plotter.add_mesh(sphere, color=color, smooth_shading=True)

    # Disegna i legami come cilindri
    for i, coord_i in enumerate(geometry.coordinates):
        for j, coord_j in enumerate(geometry.coordinates[i+1:], start=i+1):

            bond_order = bond_type_matrix[i, j]

            if only_single_bonds is True:
                bond_order = 1. if bond_order > 0. else 0.
            
            # Check if the atoms are connected by a bond if not skip the iteration
            if bond_order == 0.:
                continue

            direction = coord_i - coord_j
            center = (coord_i + coord_j) / 2
            height = np.linalg.norm(direction)
            
            # If the bond is too short skip the iteration
            if height < 1e-6:
                continue

            # Generate a vector perpendicular to the bond direction using the cross product
            # with an arbitrary axis vector. If the vectors are almost parallel to the `z`
            # axis, switch to the `y` one 

            direction /= height         # Normalize the direction vector
            if abs(direction[2]) < 0.9:
                ortho = np.cross(direction, [0, 0, 1])
            else:
                ortho = np.cross(direction, [0, 1, 0])

            ortho /= np.linalg.norm(ortho)

            # Set the spacing between cylinders representing multiple bonds
            offset = bond_radius * 1.5 

            # Plot bonds according to the computed bond order 
            # 3.0: Triple bond
            if abs(bond_order - 3) < 0.1: 
                offsets = [-offset, 0, offset]
                for k in offsets:
                    p_shift = ortho * k
                    cyl = pv.Cylinder(center=center + p_shift, direction=direction, radius=bond_radius, height=height, resolution=24)
                    plotter.add_mesh(cyl, color='lightgray', smooth_shading=True)

            # 2.0: Double bond
            elif abs(bond_order - 2) < 0.1:
                offsets = [-offset/1.5, offset/1.5]
                for k in offsets:
                    p_shift = ortho * k
                    cyl = pv.Cylinder(center=center + p_shift, direction=direction, radius=bond_radius, height=height, resolution=24)
                    plotter.add_mesh(cyl, color='lightgray', smooth_shading=True)

            # 1.5: Aromatic bond
            elif abs(bond_order - 1.5) < 0.1:
                p_shift = ortho * (offset/1.5)

                cyl = pv.Cylinder(center=center - p_shift, direction=direction, radius=bond_radius, height=height, resolution=24)
                plotter.add_mesh(cyl, color='lightgray', smooth_shading=True)

                n_segments = 6
                segment_gap = height / (2 * n_segments)
                for s in range(n_segments):
                    # centro del mini-cilindro
                    frac = (s + 0.5) / n_segments
                    segment_center = coord_i - direction * (height * frac)
                    cyl = pv.Cylinder(center=segment_center + p_shift, direction=direction, radius=bond_radius, height=segment_gap, resolution=24)
                    plotter.add_mesh(cyl, color='lightgray', smooth_shading=True)

            # 1.0: Single bond
            else:
                cyl = pv.Cylinder(center=center, direction=direction, radius=bond_radius, height=height, resolution=24)
                plotter.add_mesh(cyl, color='lightgray', smooth_shading=True)

    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = elevation
    plotter.camera.zoom(zoom)

    plotter.add_title(title, color=title_color, font_size=title_size)

    if show_axes is True:
        plotter.show_axes()
    
    if export_path:
        plotter.screenshot(export_path, transparent_background=transparent_background)
        plotter.close()
    else:
        plotter.show()
