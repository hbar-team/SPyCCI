(getting-started)=

# Getting Started

The `GES-comp-echem` package can be installed in a Conda environment with the command:
```
conda install -c greenenergystorage GES-comp-echem
```

The library can be imported in a Python script via the following syntax:

```python
import compechem
```

Alternatively, individual submodules, classes, and functions can be imported separately:

```python
from compechem import systems
from compechem.wrappers import dftbplus
from compechem.wrappers.packmol import packmol_cube
```

For a more detailed explanation of the available features in each submodule, please refer to their specific page in this [User Guide](user-guide).

---

## My first calculation

### Introduction

Let us go through the very basics of using the library. We are going to carry out a geometry optimisation on a water molecule. At the very least, you will need the geometrical structure of the system you want to study, in the form of a .xyz file. You can obtain it from available databases, or you can draw the structures yourself in programs such as [Avogadro](https://avogadro.cc/).

Below is the `water.xyz` file, containing the structure of the water molecule, which we will use in these examples:

```
3

O   0.000  -0.736   0.000  
H   1.442   0.368   0.000  
H  -1.442   0.368   0.000  
```

If you open the file in a molecular visualization software, you will notice the structure is distorted from the typical equilibrium geometry. We can then optimise the structure by utilising one of the wrappers implemented in `GES-comp-echem`. We will use [xTB](https://github.com/grimme-lab/xtb) in this example, due to its balance between accuracy and speed. The library needs the program to already be installed ([preferably via conda](https://xtb-docs.readthedocs.io/en/latest/setup.html#setup-and-installation)) and ready to go.

### Importing the library

Before starting, we need to create a Python script and import the necessary classes from the library. We need the `System` class to store the information about our water molecule, and the `XtbInput` class to define the simulation setup (Hamiltonian, parameters, solvation, etc.):

```python
from compechem.systems import System
from compechem import XtbInput
```

### Creating the System object

After importing the necessary modules, we can create our molecule, by indicating the (relative, or complete) path where the .xyz file is located:

```python
water = System("example_files/water.xyz")
```

### Creating a XtbInput object

We can now setup a wrapper object using an instance of `XtbInput`. Most of these wrappers come with sensible default options for electrochemical calculations in solution. To see all the available options, please refer to the [wrappers](API-wrappers) section.

```python
xtb = XtbInput()
```

### Carrying out the calculation

We can now carry out the calculation. We want to do a geometry optimization on our water molecule, and we want the original information for the molecule to be updated after the calculation (`inplace` flag). The syntax for this calculation is as follows:

```python
xtb.opt(water, inplace=True)
```

### Printing the results

If you want to see the data currently stored in our `System` object, simply ask for it to be printed to screen:

```python
print(water)
```

```
=========================================================
SYSTEM: water
=========================================================

Number of atoms: 3
Charge: 0
Spin multeplicity: 1

********************** GEOMETRY *************************

Total system mass: 18.0153 amu

----------------------------------------------
 index  atom    x (Å)      y (Å)      z (Å)   
----------------------------------------------
 0       O    -0.00001   -0.37937    0.00000  
 1       H     0.77221    0.18968   -0.00000  
 2       H    -0.77221    0.18969    0.00000  
----------------------------------------------

********************** PROPERTIES *************************

Electronic level of theory: XtbInput || method: gfn2 | solvent: None
Vibronic level of theory: XtbInput || method: gfn2 | solvent: None

Electronic energy: -5.070544446446 Eh
Vibronic energy: 0.002508816182 Eh
Helmholtz free energy: None Eh
Gibbs free energy: None Eh
pKa: None
```

Et voilà! You have successfully carried out a geometry optimization for the water molecule using the `GES-comp-echem` library!