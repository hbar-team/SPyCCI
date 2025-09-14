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
# Molecular visualization using VMD
Beyond allowing the user to run computational chemistry calculations, the SPyCCI package also provides a simple interface to the [Visual Molecular Dynamics (VMD) software](https://www.ks.uiuc.edu/Research/vmd/) enabling the user to render molecular structures and cube files directly from a python script. The interface is constantly updated with new features and, at the moment, supports the rendering of both molecular structures, provided either in the form of `System` objects, `.xyz` or `pdb` files, and volumetric data in the form of `.cube` files or `Cube` objects.

The core of the interface is represented by the `VMDRenderer` class. The class represents a generic rendering tool that can be created setting a resolution value, system orientation and zoom, graphical effects such as shadows, ambientocclusion and depth of field (DoF). Once created the class provides specific methods capable of generating the required renders.

<h4> Rendering a molecule from a `.xyz` file </h4>
The structure of a molecule can be rendered starting from a `.xyz` file by using the `render_system_file()` function. As an example, the following script can be used to render a simple `water.xyz` file:

```python
from spycci.tools.vmdtools import VMDRenderer

vmd = VMDRenderer(resolution=1200)
vmd.render_system_file("./water.xyz")
```

This will save a `water.bmp` image file that looks like this:

```{image} ./images/water.bmp
:alt: water.bmp
:width: 300px
:align: center
```

<h4> Rendering a molecule from a `System` object </h4>
Similarly from what shown in the case of a `.xyz` file, the `VMDRenderer` class directly supports the rendering of `System` objects. This can be done using the `render_system()` function. As an example, the following script can be used to render a benzene molecule generated from its SMILES string:

```python
from spycci.systems import System
from spycci.tools.vmdtools import VMDRenderer

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

<h4> Rendering an ORCA spin density (`.spindens.cube`) file </h4>

A spin density cube generated from orca (having the `.spindens.cube` extension) can be rendered using the dedicated `render_spin_density_cube()` function. In the following simple example, the spin density obtained form a DFT calculation has been saved and directly rendered from the `output_densities` folder:

```python
from spycci.systems import System
from spycci.engines.orca import OrcaInput
from spycci.tools.vmdtools import VMDRenderer

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

<h4> Rendering an Fukui functions (`.fukui.cube`) file </h4>

Volumetric maps of Fukui functions generated from the SPyCCI `calculate_fukui()` function, and saved in cube format (having the `.fukui.cube` extension), can be rendered using the dedicated `render_fukui_cube()` function. In the following simple example, the Fukui $f^+(r)$ function obtained form a DFT calculation on the propionaldehyde molecule has been saved and directly rendered from the `output_densities` folder:

```python
from spycci.systems import System
rom spycci.engines.xtb import XtbInput
from spycci.engines.orca import OrcaInput
from spycci.functions.fukui import calculate_fukui, CubeGrids

from spycci.tools.vmdtools import VMDRenderer

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