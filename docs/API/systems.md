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

(API-systems)=
# `spycci.systems` sub-module

## `System` class

```{eval-rst}
.. autoclass:: spycci.systems.System
    :members:
    :undoc-members:
    :private-members:
```

(API-systems-listener)=
### `System` synchronization with `MolecularGeometry` and `Properties` members

The `System` class implements a listener mechanism between each `System` instance and its `MolecularGeometry` (`self.geometry`) and `Properties` (`self.properties`) attribues. The goal is to ensure that the `System` instance is automatically notified whenever the associated attribute changes requiring updates not limited to the attribute itself. This mechanism is used under two circumstances:

- If the geometry is changed, (e.g. through the `append` method of the `MolecularGeometry` class), the properties associated with the old geometry, and stored in the `self.proprties` attribute, must be cleared. This is taken care by the `__on_geometry_change` listener of the `System` class waiting for a signal from the `MolecularGeometry` class.

- If the `STRICTNESS_LEVEL` is set to `VERY_STRICT` the properties setters of the `Properties` class must resort to an internal (private) method of the `System` class to handle the level of theory validation against the `level_of_theory_geometry`. This is taken care by the `__check_geometry_level_of_theory` listener of the `System` class waiting for a signal from the `Properties` class.

#### Synchronization of `System` with `MolecularGeometry`
To understand the synchronization mechanism of `System` with `MolecularGeometry` let us consider the following simplified structure of the `MolecularGeometry` class:

```python
class MolecularGeometry:

    def __init__(self) -> None:
        self.__system_reset: System.__on_geometry_change = None
    
    def __add_system_reset(self, listener: System.__on_geometry_change) -> None:
        self.__system_reset = listener
    
    def __call_system_reset(self) -> None:
        if self.__system_reset is not None:
            self.__system_reset()
```

Within `MolecularGeometry`, there is a private attribute called `__system_reset` that stores a callback function that, as typing suggest, will be the listener function implemented in the `System` class. The method `__add_system_reset` allows registering this callback, while `__call_system_reset` triggers it when a change occurs. To understand how all this comes together let's consider the following simplified structure for the `System` class:

``` python
class System:

    def __init__(self, geometry: MolecularGeometry) -> None:
        self.__geometry: MolecularGeometry = deepcopy(geometry)
        self.__geometry._MolecularGeometry__add_system_reset(self.__on_geometry_change)
    
    def __on_geometry_change(self) -> None:
        self.properties = Properties()
```

`System` receives a `MolecularGeometry` instance during initialization and registers its private method `__on_geometry_change` as a callback through `__add_system_reset`. Please notice how in this case variable mangling is used explicitly to avoid exposing to the user the `__add_system_reset` of the `MolecularGeometry` avoidning as such unintentional listener inactivation. Whenever `MolecularGeometry` calls `__call_system_reset`, the `System` is notified and can update its properties, for example by recreating the `Properties` object. This approach implements a simple observer pattern, ensuring that any geometry modification is automatically propagated to the system without manual intervention. During the operation no information needs to be transferred between the two classes and, as such, the `__on_geometry_change` method does not take any argument nor returns any value.

Furthermore, to avoid undefined behaviors in the form of unwanted calls to clearing of `MolecularGeometry` instances created by `deepcopy`, a `__deepcopy__` overload is defined within the `MolecularGeometry` clearing the `__system_reset` callback inhibiting as such undesired clearing operations:

```python
def __deepcopy__(self, memo) -> MolecularGeometry:
    cls = self.__class__
    obj = cls.__new__(cls)
    memo[id(self)] = obj

    for attr_name, attr_value in self.__dict__.items():
        setattr(obj, attr_name, deepcopy(attr_value, memo))
    
    obj.__system_reset = None

    return obj
```

#### Synchronization of `System` with `Properties`
The synchronization mechanism implemented between the `System` and the `Properties` class is analogous to the one already discussed for `MolecularGeometry`. Within the `Properties` class a  `__check_geometry_level_of_theory` private attribute stores a callback function that, as typing suggest, will be the listener function implemented in the `System` class. The method `__add_check_geometry_level_of_theory` allows registering this callback, while `__call_check_geometry_level_of_theory` triggers it when a level of theory validation is required. 

```python
class Properties:
    
    def __init__(self):
        self.__check_geometry_level_of_theory: System.__check_geometry_level_of_theory = None
    
    def __add_check_geometry_level_of_theory(self, listener: System.__check_geometry_level_of_theory) -> None:
        self.__check_geometry_level_of_theory = listener
    
    def __call_check_geometry_level_of_theory(self, engine: Union[Engine, str]) -> None:
        if self.__check_geometry_level_of_theory is not None:
            self.__check_geometry_level_of_theory(engine)
```

`System` sets a `Properties` instance during initialization and registers its private method `__check_geometry_level_of_theory` as a callback through `__add_check_geometry_level_of_theory`. Whenever `Properties` calls `__call_check_geometry_level_of_theory`, the `System` is notified and checks the level of theory provided by the `Properties` object and the `level_of_theory_geometry` of the `MolecularGeometry` class. Due to the need of checking the incoming string, the `__check_geometry_level_of_theory` is set to accept a single argument and return nothing.

```python
class System:

    def __init__(self) -> None:
        self.__properties: Properties = Properties()
        self.__properties._Properties__add_check_geometry_level_of_theory(self.__check_geometry_level_of_theory)

    def __check_geometry_level_of_theory(self, level_of_theory: str) -> None:
        if self.geometry.level_of_theory_geometry is not None:
            if level_of_theory != self.geometry.level_of_theory_geometry:
                raise RuntimeError("Mismatch between the user-provided level of theory and the one used to set geometry")
```

Also in this case, a `__deepcopy__` overload is implemented in `Properties` to remove listener link when a `Properties` instance is separated from its original owner.

---

## `ReactionPath` class

```{eval-rst}
.. autoclass:: spycci.systems.ReactionPath
    :members:
```

---

## `Ensemble` class

```{eval-rst}
.. autoclass:: spycci.systems.Ensemble
    :members:
```
