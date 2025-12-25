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

(Guide-geometry)=
# The `MolecularGeometry` class

The `MolecularGeometry` class provides a unified interface for storing, manipulating, and analyzing the 3D geometry of molecules or molecular aggregates. The class is designed as the internal geometric backbone of a `System` object, but can also be used as a standalone molecular geometry handler.

A `MolecularGeometry` instance contains:
- The list of elements composing the molecule (`atoms`)
- The corresponding Cartesian coordinates in Ångström (`coordinates`)
- Metadata such as the level of theory used to obtain the geometry (`level_of_theory_geometry`)
- Derived geometric properties such as center of mass, moments of inertia, and rotational constants.

The class emphasizes data consistency:
- All structural quantities (inertia tensor, rotor type, etc.) are automatically invalidated whenever atoms or coordinates are modified and automatically computed when not set.
- When used as part of a `System`, it notifies the parent via an internal geometry-change listener the architecture of which is [described in the API](API-systems-listener).

## Creating a `MolecularGeometry` object

The `__init__` method of the `MolecularGeometry` class does not accept any argument; as such, when created, an instance of the `MolecularGeometry` class is empty and no atoms are described. These can be added using the `append` method by providing each atom with its own space coordinates. As an example, the following code can be used to define the geometry of the water molecule:

```{code-cell} python
from spycci.core.geometry import MolecularGeometry

WATER = [
    ["O", -5.02534, 1.26595, 0.01097],
    ["H", -4.05210, 1.22164, -0.01263],
    ["H", -5.30240, 0.44124, -0.42809],
]

# Creating a MolecularGeometry onject
geom = MolecularGeometry()
for l in WATER:
    geom.append(l[0], l[1::])

# Print some attributes
print(f"Number of atoms: {geom.atomcount}")
print(f"Atoms list: {geom.atoms}")
```

This, despite being the simplest possible approach, is rarely used and pre-made `classmethods` are provided to process the most common source of chemical structures like `xyz` files and SMILES strings.

### Loading data from an `.xyz` file

A `MolecularGeometry` object can be initialized in various ways the most common of which is that of providing a molecular structure via a `.xyz` file. This can be done using the `from_xyz()` classmethod. In the following example a molecular geometry object, containing the structure of the water molecule, is loaded from the `water.xyz` file and some of its properies are printed:

```{code-cell} python
from spycci.core.geometry import MolecularGeometry

# Create a `MolecularGeometry` object for a neutral water molecule from a `.xyz` file
geom = MolecularGeometry.from_xyz("../example_files/water.xyz")

# Print some attributes
print(f"Number of atoms: {geom.atomcount}")
print(f"Atoms list: {geom.atoms}")
```

Please notice how the sytax is very similar to the one adopted in the previous section when creating `System` objects. This is not a case considering that the `from_xyz()` classmethod of the `System` class simply wraps the one from the `MolecularGeometry` class and passes the resulting geometry object to the `__init__` method of the `System` class. This will also be the case from the next example about SMILES.

### Creating a `System` object form a SMILES string

Another convenient way to initialize a `MolecularGeometry` object is that of using SMILES strings. This can be done using the `form_smiles` class method according. As an example the geometry of the methane molecule can be created following the syntax:


```{code-cell} python
from spycci.core.geometry import MolecularGeometry

geom = MolecularGeometry.from_smiles("C")

# Print some attributes
print(f"Number of atoms: {geom.atomcount}")
print(f"Atoms list: {geom.atoms}")
```

The class constructor automatically adds implicit hydrogens to the structure, embeds it in the 3D space and runs a rough geometry optimization using either the `MMFF`s or the `UFF` force fields. Additional optional keywords that can be used to configure the SMILES translation process can be found in the [`MolecularGeometry` object API](core-geometry-API). 

## Editing molecular data

Once a `MolecularGeometry` object is created, the list of atoms and relative coordinates can be accessed using the `atoms` and `coordinates` properties getters according to the syntax:

```{code-cell} python
# Creating an example system
from spycci.core.geometry import MolecularGeometry
geom = MolecularGeometry.from_smiles("C")

# Access atoms and coordinates using the property getter
atoms = geom.atoms
coordinates = geom.coordinates

# Print atoms and coordinates
for a, c in zip(atoms, coordinates):
    print(f"{a}\t  {c[0]:.3f}\t  {c[1]:.3f}\t  {c[2]:.3f}")
```

If the user wants to set new atom labels or change coordinate values, he can do so using the provided `set_atoms()` and `set_coordinates()` setter functions. As an example:

```{code-cell} python
# Creating an example system
from spycci.core.geometry import MolecularGeometry
geom = MolecularGeometry.from_smiles("C")

# Print atoms and coordinates
for a, c in zip(atoms, coordinates):
    print(f"{a}\t  {c[0]:.3f}\t  {c[1]:.3f}\t  {c[2]:.3f}")

# Change atom list using the `set_atoms()` setter
geom.set_atoms(["Sn", "H", "H", "H", "H"])

# Print atoms and coordinates
for a, c in zip(atoms, coordinates):
    print(f"{a}\t  {c[0]:.3f}\t  {c[1]:.3f}\t  {c[2]:.3f}")
```

Beware that directly trying to change the properties by direct assignment will result in an exception. For example:

```{code-cell} python
# Creating an example system
from spycci.core.geometry import MolecularGeometry
geom = MolecularGeometry.from_smiles("C")

# Try to change the atom list by direct assignment
try:
    geom.atoms = ["Sn", "H", "H", "H", "H"]

except Exception as e:
    print(f"Directly assigning protected property resulted in the exception: {e}")
```

At the same time, trying to directly modify the protected variables by reference will be ineffective since the values (even the mutable ones) are returned as a `deepcopy` of the protected object. For example, looking at the following code:

```{code-cell} python
# Creating an example system
from spycci.core.geometry import MolecularGeometry
geom = MolecularGeometry.from_smiles("C")

# Print atoms and coordinates
print("Before:")
for a, c in zip(atoms, coordinates):
    print(f"{a}\t  {c[0]:.3f}\t  {c[1]:.3f}\t  {c[2]:.3f}")
print("\n")

# Try to change an atom by reference
geom.atoms[0] = "Sn"

# Print atoms and coordinates
print("After:")
for a, c in zip(atoms, coordinates):
    print(f"{a}\t  {c[0]:.3f}\t  {c[1]:.3f}\t  {c[2]:.3f}")
```

we can see how the atom list is unchanged. This because the `atoms` property getters returns a `deepcopy` of the atoms list and, as such, the `geom.atoms[0] = "Sn"` lines affect only the temoprary copy and not the inner value of the `MolecularGeometry` instance.

This behavior is expected for all mutable properties of the `MolecularGeometry` and ensures that data cannot be altered without the imput triggering proper validation methods ensuring data validity and consistency; i.e If the `MolecularGeometry` object is part of a `System` object, a call to `set_atoms()` will change the system definition an, as such, clear the system properties.

## Structural properties

The `MolecularGeometry` class also implements some internal properties directly computed from its definition:

| Property | Description | Return type | Setter |
|-----------|--------------|--------------|---------|
| `atomcount` | Total number of atoms in the geometry. | `int` | ❌ (Computed) |
| `atoms` | List of element symbols for each atom. | `list[str]` | `set_atoms()` |
| `atomic_numbers` | List of atomic numbers for each atom. | `list[int]` | ❌ (Computed) |
| `coordinates` | Cartesian coordinates (in Å) for each atom | `list[np.ndarray[float]]` | `set_coordinates()` |
| `level_of_theory_geometry` | Level of theory used to obtain the geometry  | `str or None` | direct assignment |
| `mass` | Total molecular mass (in amu). | `float` | ❌ (Computed) |
| `center_of_mass` | Coordinates of the molecular center of mass (Å). | `list[float]` | ❌ (Computed) |
| `inertia_tensor` | Full 3×3 inertia tensor (amu·Å²). | `np.ndarray` (3×3) | ❌ (Computed) |
| `inertia_eigvals` | Principal moments of inertia in amu·Å². | `tuple[float, float, float]` | ❌ (Computed) |
| `inertia_eigvecs` | Principal axes of rotation (unit vectors). | `np.ndarray` (3×3) | ❌ (Computed) |
| `rotor_type` | Type of rigid rotor. | `str` | ❌ (Computed) |
| `rotational_constants` | Rotational constants in cm⁻¹ and MHz. | `tuple[np.ndarray, np.ndarray]` | ❌ (Computed) |

