---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(Guide-Molecular-Visualization)=
# Molecular visualization 
Beyond allowing the user to run computational chemistry calculations, the SPyCCI package also provides the user with a simple interface to some visualization tools that can be used to examine molecular structures, computed values and also volumetric data.

The SPyCCI library provides an internal molecular viewer and wrappers around third party softwares like VMD and mogli. All the molecular visualization tools are available under the `spycci.graphicaltools` sub-module.

## Molecular visualization using the internal viewer
The SPyCCI library provides a simple molecular viewer based on pyvista. The tool can be found in the `spycci.graphicaltools.moleculeviewer` sub-module in the form of the `show_molecule` function.

To use the `show_molecule` function the user needs to simply provide the function with the geometry to be visualized in the form of either a `System` or `MolecularGeometry` object. The tool will automatically open a window allowing the user to interact and rotate the molecule. As an example, the following script can be used to visualize the acetone molecule:

```python
from spycci.systems import System
from spycci.graphicaltools.moleculeviewer import show_molecule

mol = System.from_smiles("acetone", "CC(=O)C")
show_molecule(mol)
```

```{image} ./images/acetone.png
:alt: water.bmp
:width: 600px
:align: center
```

Camera orientation can be set via the `azimuth` and `elevation` arguments while the zoom can set using the `zoom` keyword. As an example, the previous view can be modified as follows:

```python
from spycci.systems import System
from spycci.graphicaltools.moleculeviewer import show_molecule

mol = System.from_smiles("acetone", "CC(=O)C")
show_molecule(mol, azimuth=10, elevation=180, zoom=1.4)
```

```{image} ./images/acetone_rotated.png
:alt: water.bmp
:width: 600px
:align: center
```

Besides visualizing structures, the internal viewer can also be used to represent scalar properties associated with each atom in the molecule. This can be done automatically by passing to the function a list of colors to be used in the representation of each atoms. To do so, the user can define a colormap function or, more conveniently, use one of the colormaps provided in the `spycci.graphicaltools.colormaps` sub-module. The colormap will handle the conversion of scalar values to colors that can be directly used by the viewer. As an example, conisder the following code in which Mulliken charges are computed using orca, converted in colorscale using the `RdBu` colormap and represented by the viewer. The obtained scene is finally rendered and saved to a `.png` image file:

```python
from spycci.systems import System
from spycci.engines.orca import OrcaInput
from spycci.graphicaltools.colormaps import RdBu
from spycci.graphicaltools.moleculeviewer import show_molecule

# Define a System representing the acetone molecule
mol = System.from_smiles("acetone", "CC(=O)C")

# Run a geometry optimization using orca
orca = OrcaInput()
orca.opt(mol, ncores=4, inplace=True)

# Convert Mulliken charges values to color scale usinf the RdBu colormap
colors = RdBu(mol.properties.mulliken_charges)

# Render the molecule using the RdBu diverging colormap
show_molecule(
    mol,
    atoms_colors = colors,
    azimuth=10,
    elevation=180,
    zoom=1.4,
    title="Acetone - Mulliken charges",
    export_path="acetone_charges.png",
)
```

```{image} ./images/acetone_charges.png
:alt: water.bmp
:width: 600px
:align: center
```

### Built-in colormaps 

To help the user in visualizing scalar data, a small collection of colormap function has been defined in the `spycci.graphicaltools.colormaps` sub-module. The module is built around two type of colormaps: A `DivergingColormap` function that provides linearly diverging color shades centered around a central value, and a `PolynomialColormap` implementing a polynomial representation of each RGB channel value based on the input variable. From these two basic definition a series of built-in colormaps has been defined and represented in what follows:

```{code-cell} python
:tags: ["remove-input"]

import numpy as np
import inspect
import matplotlib.pyplot as plt
from spycci.graphicaltools.colormaps import RdBu, RdYlBu, PiYG, Jet, Turbo, Viridis, Plasma

def show_colormaps(cmaps, n=256, clims=None):
    """
    Display multiple custom colormaps stacked vertically.

    Parameters
    ----------
    cmaps : list of tuples
        A list of (colormap_function, title) pairs.
    n : int
        Number of discrete samples for visualization.
    clims : tuple or None
        Min/max range of the colormap. If None, automatically selected:
        - (-1, 1) for diverging colormaps (e.g. RdBu)
        - (0, 1) for sequential colormaps
    """
    fig, axes = plt.subplots(len(cmaps), 1, figsize=(8, 1. * len(cmaps)))

    if len(cmaps) == 1:
        axes = [axes]

    for ax, (cmap_func, title) in zip(axes, cmaps):
        # Automatically detect diverging colormaps
        if clims is None:
            if "symmetric" in inspect.signature(cmap_func).parameters:
                vmin, vmax = -1, 1
            else:
                vmin, vmax = 0, 1
        else:
            vmin, vmax = clims

        values = np.linspace(vmin, vmax, n)
        colors = np.array(cmap_func(values))
        colors = np.clip(colors, 0, 1)

        ax.imshow([colors], aspect='auto', extent=[vmin, vmax, 0, 1])
        ax.set_yticks([])
        ax.set_xlabel(title, fontsize=11)
        ax.tick_params(axis='x', labelsize=9)

        # Centered x-axis for diverging maps
        if vmin < 0 < vmax:
            ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


cmaps = [
    (RdBu, "RdBu"),
    (RdYlBu, "RdYlBu"),
    (PiYG, "PiYG"),
    (Jet, "Jet"),
    (Turbo, "Turbo"),
    (Viridis, "Viridis"),
    (Plasma, "Plasma"),
]

show_colormaps(cmaps)
```

## Molecular visualization using VMD
Beyond allowing the user to run computational chemistry calculations, the SPyCCI package also provides a simple interface to the [Visual Molecular Dynamics (VMD) software](https://www.ks.uiuc.edu/Research/vmd/) enabling the user to render molecular structures and cube files directly from a python script. The interface is constantly updated with new features and, at the moment, supports the rendering of both molecular structures, provided either in the form of `System` objects, `.xyz` or `pdb` files, and volumetric data in the form of `.cube` files or `Cube` objects.

The core of the interface is represented by the `VMDRenderer` class. The class represents a generic rendering tool that can be created setting a resolution value, system position, orientation and zoom, graphical effects such as shadows, ambientocclusion and depth of field (DoF). Once created the class provides specific methods capable of generating the required renders.

:::{admonition} Notes about cube files
:class: info
The `.cube` format processed by SPyCCI is the [Gaussian cube file format](https://paulbourke.net/dataformats/cube/). The file is composed by a header and a body of volumetric data. The header is structured as:
- Two comment lines.
- A line listing the number of atoms included in the file followed by the position of the origin of the volumetric data.
- Three lines listing the number of voxels along each axis (`x`, `y`, `z`) followed by the axis vector. The length of each vector is the length of the side of the voxel thus allowing non cubic volumes. If the sign of the number of voxels in a dimension is positive then the units are Bohr, if negative then Angstroms.
- A section with a number of lines equal to the number of atoms in the molecule and consisting of 5 numbers: the first is the atom number, the second is the charge (this field is often used for other data types), and the last three are the `x`, `y`, `z` coordinates of the atom center.

The body of volumetric data is represented by a series of lines listing floating point values for each voxel. The data is usually arranged in columns of 6 values generated by looping on the voxel coordinates with `x` axis as the outer loop and `z` as the most inner one.
:::


<h4> Rendering a molecule from a <code>.xyz</code> file </h4>

The structure of a molecule can be rendered starting from a `.xyz` file by using the `render_system_file()` function. As an example, the following script can be used to render a simple `water.xyz` file:

```python
from spycci.graphicaltools.vmdtools import VMDRenderer

vmd = VMDRenderer(resolution=1200)
vmd.render_system_file("./water.xyz")
```

This will save a `water.bmp` image file that looks like this:

```{image} ./images/water.bmp
:alt: water.bmp
:width: 300px
:align: center
```

<h4> Rendering a molecule from a <code>System</code> object </h4>

Similarly from what shown in the case of a `.xyz` file, the `VMDRenderer` class directly supports the rendering of `System` objects. This can be done using the `render_system()` function. As an example, the following script can be used to render a benzene molecule generated from its SMILES string:

```python
from spycci.systems import System
from spycci.graphicaltools.vmdtools import VMDRenderer

mol = System.from_smiles("benzene", "c1ccccc1")

renderer = VMDRenderer(resolution=1200)
renderer.scale = 1.5
renderer.xyx_rotation = [-45., 0., 0.]

renderer.render_system(mol)
```

Please notice how the `scale` argument has been set to `1.5` to zoom while a rotation of 45° has been applied to the `x` axis. The output of such a script is a `benzene_0_1.bmp` file:

```{image} ./images/benzene_0_1.bmp
:alt: benzene_0_1.bmp
:width: 400px
:align: center
```


:::{admonition} Notes about rotations
:class: info
The current implementation of the `VMDRendered` class uses an XYX rotation sequence, following the convention of proper Euler angles. Unlike intrinsic rotations tied to the molecular frame, these rotations are applied around the camera (screen) axes, which change orientation after each step. As a result, the second application of the X-axis rotation occurs around a different orientation than the first, due to the intermediate Y-axis rotation. This sequence allows representation of arbitrary 3D orientations, despite the repeated axis label.
:::

<h4> Rendering a generic <code>.cube</code> file </h4>

Rendering of a generic cube file can be achieved using the `render_cube_file()` function of the `VMDRenderer` class. The function, that can be called by simply providing the name of the cube file, can be set to render positive and negative regions of volumetric data with a user specified `isovalue`. If the isovalue is not explicilty indicated a default value will be computed as the 20% of the maximum voxel value (in absolute values). Additional features such as the color of the positive and negative reagions, whether to show the negative part of the cube or the filename of the output file, can be set by the user using the built in keywords.

As an example, the 6th molecular orbital for the formaldehyde molecule, generated by orca as a `input.mo6a.cube` file. Can be simply rendered using the code:

```python
from spycci.graphicaltools.vmdtools import VMDRenderer

renderer = VMDRenderer(resolution=1200)
renderer.scale = 1.0
renderer.xyx_rotation = [120.0, 20.0, 0.0]

renderer.render_cube_file(
    f"./input.mo6a.cube",
    positive_color=32,
    negative_color=23,
    show_negative=True,
)
```

resulting in the following `input.mo6a.bmp` image file:

```{image} ./images/input.mo6a.bmp
:alt: input.mo6a.bmp
:width: 450px
:align: center
```

:::{admonition} Notes about ORCA's molecular orbitals (MOs) cube files
:class: warning
The cube files generated by ORCA when exporting molecular orbitals differs from the standard structure of the Gaussian cube file format. The number of atoms is represented by a negative number and an additional line is added after the atoms coordinates section. The differences in formatting are taken into account by SPyCCI that automatically converts this format to a regular Gaussian cube for visualization with `vmd`.
:::

<h4> Rendering volumetric data from a <code>Cube</code> object </h4>

Similarly from what shown in the case of a `.cube` file, the `VMDRenderer` class directly supports the rendering of `Cube` objects. This can be done using the `render_cube()` function in complete analogy with the syntax discussed before for the `render_cube_file()` function. The only difference in this case is the requirement for the user to provide a filename since, differently from what seen in the case of a `System` object, a `Cube` object has not assigned name.

As an example, the previous example about plotting the 6th molecular orbital of the formaldehyde molecule from a `input.mo6a.cube` cube file can be revritten as follows:

```python
from spycci.tools.cubetools import Cube
from spycci.graphicaltools.vmdtools import VMDRenderer

renderer = VMDRenderer(resolution=400)
renderer.scale = 1.0
renderer.xyx_rotation = [120.0, 20.0, 0.0]

cube = Cube.from_file("./input.mo6a.cube")

renderer.render_cube(
    cube,
    f"./formaldehyde_MO6a.bmp",
    positive_color=32,
    negative_color=23,
    show_negative=True,
)
```

This will generate a `formaldehyde_MO6a.bmp` file identical to the one shown in the previous example.

<h4> Rendering an ORCA spin density (<code>.spindens.cube</code>) file </h4>

A spin density cube generated from orca (having the `.spindens.cube` extension) can be rendered using the dedicated `render_spin_density_cube()` function. In the following simple example, the spin density obtained form a DFT calculation has been saved and directly rendered from the `output_densities` folder:

```python
from spycci.systems import System
from spycci.engines.orca import OrcaInput
from spycci.graphicaltools.vmdtools import VMDRenderer

mol = System.from_smiles("benzene", "c1ccccc1", charge=1, spin=2)

orca = OrcaInput(method="B3LYP", basis_set="def2-TZVP", ncores=4)
orca.spe(mol, save_cubes=True, inplace=True)

renderer = VMDRenderer(resolution=1200)
renderer.scale = 1.5
renderer.xyx_rotation = [-45.0, 0.0, 0.0]

renderer.render_spin_density_cube(
    "./output_densities/benzene_1_2_orca_B3LYP_def2-TZVP_vacuum_spe.spindens.cube",
    filename="spindensity.bmp"
)
```

The result is a `spindensity.bmp` (the name of the output has been manually set using the `filename` option):

```{image} ./images/spindensity.bmp
:alt: spindensity.bmp
:width: 450px
:align: center
```

<h4> Rendering an Fukui functions (<code>.fukui.cube</code>) file </h4>

Volumetric maps of Fukui functions generated from the SPyCCI `calculate_fukui()` function, and saved in cube format (having the `.fukui.cube` extension), can be rendered using the dedicated `render_fukui_cube()` function. In the following simple example, the Fukui $f^+(r)$ function obtained form a DFT calculation on the propionaldehyde molecule has been saved and directly rendered from the `output_densities` folder:

```python
from spycci.systems import System
rom spycci.engines.xtb import XtbInput
from spycci.engines.orca import OrcaInput
from spycci.functions.fukui import calculate_fukui, CubeGrids

from spycci.graphicaltools.vmdtools import VMDRenderer

mol = System.from_smiles("propionaldehyde", "CCC=O")
xtb = XtbInput()

xtb.opt(mol)

orca = OrcaInput(method="PBE", basis_set="def2-SVP", ncores=4)
calculate_fukui(mol, orca, cube_grid=CubeGrids.FINE)

renderer = VMDRenderer(resolution=1200)
renderer.scale = 1.4
renderer.xyx_rotation = [-100.0, 180.0, 90.0]

renderer.render_fukui_cube(
    "./output_densities/propionaldehyde_orca_PBE_def2-SVP_vacuum_Fukui_plus.fukui.cube",
    isovalue=0.01,
    show_negative=True,
    filename="fukui_plus.bmp",
)
```

The output of the renderer call is a `fukui_plus.bmp` image file:

```{image} ./images/fukui_plus.bmp
:alt: fukui_plus.bmp
:width: 450px
:align: center
```

Similarly condensed Fukui values, saved in the charge column of the `.fukui.cube` file, can be rendered using the `render_condensed_fukui()` function according to the syntax:

```python
renderer.render_condensed_fukui(
    "./output_densities/propionaldehyde_orca_PBE_def2-SVP_vacuum_Fukui_plus.fukui.cube",
    filename="condensed_fukui_plus.bmp",
)
```
that outputs the following `condensed_fukui_plus.bmp` image file:

```{image} ./images/condensed_fukui_plus.bmp
:alt: condensed_fukui_plus.bmp
:width: 450px
:align: center
```