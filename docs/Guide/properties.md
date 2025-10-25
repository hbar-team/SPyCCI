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

(Guide-properties)=
# Properties handling
After running calculations or calling dedicated [funtions](Guide-functions) the obtained results are stored in the `properties` attribute, (an instance of the `Properties` class), of the `System` object itself. 

The `Properties` class is designed to store and manage computed properties (e.g. energies, vibrational data, pKa, etc.) and label them with the **level of theory** used to compute them. To do so, each property is tagged internally with either an **electronic** or **vibrational** level of theory. This association is used to ensure consistency and prevent mixing data computed with different theoretical methods. This is done under the strict assumption that, when analyzing or comparing computed data, it's crucial to know under which assumptions (level of theory) each property was derived. Furthermore, each system is labelled with a **geometric** level of theory that further put into context the origin of the geometry at hand and the consistency among the levels of theory employed.

The `Properties` class automatically:

- Tracks levels of theory used for each type of property
- Ensures consistency depending on a user-defined strictness level
- Prevents undefined behavior when combining incompatible data

The behavior of the class is governed by the `STRICTNESS_LEVEL` setting (defined in `spycci.config`), which defines how strictly levels of theory must match. The `STRICTNESS_LEVEL` can assume different values defined in the `spycci.config.StrictnessLevel` enumeration. At the moment, three levels have been implemented:

- `NORMAL`: Electronic and vibrational levels of theory may differ (**default**). Consistency is ensured only within data of the same type (e.g. all electronic properties must be computed at the same electronic level of theory that can be different from the one used for vibrational data).
- `STRICT`: Electronic and vibrational levels of theory must match. A mismatch causes **automatic clearing** of conflicting properties and a **warning** is issued in the logger. An **exception is raised only** for *mixed* properties like `pKa`, where both levels must be consistent at once.
- `VERY_STRICT`: Geometric, Electronic and vibrational levels of theory must match. A mismatch between electronic and vibronic level of theory results in **automatic clearing** of conflicting properties and a **warning** is issued in the logger. A mismatch between the geometry level of theory and either of the electronic or vibrational levels of theory result in an `RuntimeError` exception. In the case of a `None` geometry level of theory, no exception is raised.

Levels of theory are available as read-only `Properties` class arguments under the names `level_of_theory_electronic` and `level_of_theory_vibrational`. The `level_of_theory_geometry` is available as read-only argument of the `MolecularGeometry` class. Instances of both class are available in a `System` object under the attributes `geometry` (type `MolecularGeometry`) and `properties` (type `Properties`).

The full list of property implemented is the following:

| Getter Function               | Setter Function                        | Level(s) of Theory       | Type                         |
|-------------------------------|----------------------------------------|--------------------------|------------------------------|
| `electronic_energy`           | `set_electronic_energy`                | Electronic               | `float`                      |
| `free_energy_correction`      | `set_free_energy_correction`           | Vibrational              | `float`                      |
| `gibbs_free_energy`           | *(computed property)*                  | Electronic + Vibrational | `float`                      |
| `pka`                         | `set_pka`                              | Electronic + Vibrational | `pKa`                        |
| `mulliken_charges`            | `set_mulliken_charges`                 | Electronic               | `List[float]`                |
| `mulliken_spin_populations`   | `set_mulliken_spin_populations`        | Electronic               | `List[float]`                |
| `condensed_fukui_mulliken`    | `set_condensed_fukui_mulliken`         | Electronic               | `Dict[str, List[float]]`     |
| `hirshfeld_charges`           | `set_hirshfeld_charges`                | Electronic               | `List[float]`                |
| `hirshfeld_spin_populations`  | `set_hirshfeld_spin_populations`       | Electronic               | `List[float]`                |
| `condensed_fukui_hirshfeld`   | `set_condensed_fukui_hirshfeld`        | Electronic               | `Dict[str, List[float]]`     |
| `vibrational_data`            | `set_vibrational_data`                 | Vibrational              | `VibrationalData`            |

All property setters automatically validate the level of theory used via an `Engine` object or string identifier. In `STRICT` mode, this validation can trigger property resets between properties belonging to different levels of theory.

## Accessing properties
A given property can be accessed by name (that matches the getter function), As an example, the following code can be used to access the electronic energy of a given `System` object:

```{code-cell} python
from spycci.systems import System

# Load an example system
water = System.from_json("../example_files/water.json")

# Read properties
el_energy = water.properties.electronic_energy
el_lot = water.properties.level_of_theory_electronic

# Print properties
print(f"Electronic energy: {el_energy} Eh")
print(f"Electronic level of theory: {el_lot}")
```

## Setting properties
A property can be set (usually this is done internally by SPyCCI) using the provided setter that besides the value to be set requires the user to provide the level of theory at which the value has been obtained. Just to provide an artifical example:

```{code-cell} python
from spycci.systems import System
from spycci.engines.orca import OrcaInput

# Load an example system
methane = System.from_smiles("methane", "C")

# Define a dummy engine to be used as a level of theory
orca = OrcaInput()

# Set the electronic energy
methane.properties.set_electronic_energy(-10.5, orca)

# Read properties (to check set values)
el_energy = methane.properties.electronic_energy
el_lot = methane.properties.level_of_theory_electronic

# Print properties
print(f"Electronic energy: {el_energy} Eh")
print(f"Electronic level of theory: {el_lot}")
```

Beware that, even in the case of mutable objects, the properties are returned as a `deepcopy` of the protected attributes. As such, trying to modify protected values by reference, (e.g. using the property getter), will be ineffective as the edit will be applied only to the returned temporary copy and not the stored data. To edit the data the user must pass through the defined setter functions.

### Use of the `STRICTNESS_LEVEL` option
As anticipated before, when setting properties, sanity checks are enforced to ensure that properties coming from different calculations have been computed at a consistent level of theory.

In `NORMAL` mode, `level_of_theory_electronic` and `level_of_theory_vibrational` can differ. As such the following code will run without any warning or data loss:

```{code-cell} python
from spycci.systems import System
from spycci.engines.xtb import XtbInput
from spycci.engines.orca import OrcaInput

# Load an example system
methane = System.from_smiles("methane", "C")

# Define dummy engines to be used as levels of theory
xtb = XtbInput()
orca = OrcaInput()

# Set the electronic energy
methane.properties.set_electronic_energy(-10.5, orca)
methane.properties.set_free_energy_correction(-0.2, xtb)

# Read properties
electronic_energy = methane.properties.electronic_energy
free_energy_correction = methane.properties.free_energy_correction

# Read levels of theory
el_lot = methane.properties.level_of_theory_electronic
vib_lot = methane.properties.level_of_theory_vibrational

# Print
print(f"Electronic energy: {electronic_energy} Eh")
print(f"Free energy correction: {free_energy_correction} Eh")
print()
print(f"Electronic level of theory: {el_lot}")
print(f"Vibronic level of theory: {vib_lot}")
```

In `STRICT` mode, `level_of_theory_electronic` and `level_of_theory_vibrational` cannot differ. As such the following code will run throwing a warning and clearing the previously stored incompatible data:

```{code-cell} python
from spycci.systems import System
from spycci.engines.xtb import XtbInput
from spycci.engines.orca import OrcaInput

# Set STRICT mode
import spycci.config
spycci.config.STRICTNESS_LEVEL = spycci.config.StrictnessLevel.STRICT

# Load an example system
methane = System.from_smiles("methane", "C")

# Define dummy engines to be used as levels of theory
xtb = XtbInput()
orca = OrcaInput()

# Set the electronic energy
methane.properties.set_electronic_energy(-10.5, orca)
methane.properties.set_free_energy_correction(-0.2, xtb)

# Read properties
electronic_energy = methane.properties.electronic_energy
free_energy_correction = methane.properties.free_energy_correction

# Read levels of theory
el_lot = methane.properties.level_of_theory_electronic
vib_lot = methane.properties.level_of_theory_vibrational

# Print
print(f"Electronic energy: {electronic_energy}")
print(f"Free energy correction: {free_energy_correction} Eh")
print()
print(f"Electronic level of theory: {el_lot}")
print(f"Vibronic level of theory: {vib_lot}")
```

