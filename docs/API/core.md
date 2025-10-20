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

(API-core)=
# `spycci.core` module

The `spycci.core` module is comprised of many different submodules used to define the inner workings and basic components of the `spycci` library:

* `spycci.core.base`: Encoding the basic definition of `Engine`.
* `spycci.core.decorators`: Encoding decorators (e.g. `@protect`) useful in tuning the properties of the library objects.
* `spycci.core.dependency_finder`: Encoding the mechanics followed by `spycci` in looking for software and dependencies.
* `spycci.core.geometry`: Encoding the concept of `MolecularGeometry`
* `spycci.core.properties`: Encoding the concept of `Properties`
* `spycci.core.spectroscopy`: Encoding all the objects and functions relevant to the field of spectroscopy.


## The `spycci.core.base` sub-module

```{eval-rst}
.. autoclass:: spycci.core.base.Engine
    :members:
    :undoc-members:
    :private-members:
```

---

## The `spycci.core.dependency_finder` sub-module

```{eval-rst}
.. automodule:: spycci.core.dependency_finder
    :members:
    :undoc-members:
    :private-members:
```

---

(core-geometry-API)=
## The `spycci.core.geometry` sub-module

```{eval-rst}
.. automodule:: spycci.core.geometry
    :members:
    :undoc-members:
    :private-members:
```

---

## The `spycci.core.properties` sub-module

```{eval-rst}
.. autoclass:: spycci.core.properties.Properties
    :members:
    :undoc-members:
    :private-members:
```
### The `pKa` property

```{eval-rst}
.. autoclass:: spycci.core.properties.pKa
    :members:
    :undoc-members:
    :private-members:
```

---

## The `spycci.core.decorators` sub-module

This module provides decorators useful in the definition of function behavior.

```{eval-rst}
.. automodule:: spycci.core.decorators
    :members:
    :undoc-members:
    :private-members:
```

### The `@protect` decorator
The `@protect` decorator can be applied to both regular and generator functions and ensures that returned or yielded objects are deeply copied and protected from modification.

Internally, `_find_depth` determines the nesting level of iterable objects, while `_apply_protection` recursively converts mutable containers (like lists) into tuples and sets NumPy arrays as non-writable (`array.flags.writeable = False`). Scalar values or non-iterables are simply deep-copied.

When applied to a function, `@protect` intercepts its output:

* For regular functions, the return value is replaced with a protected version.
* For generator functions, each yielded element is individually protected before being returned to the caller.

This mechanism prevents accidental in-place modifications of data structures passed between components, improving immutability and reproducibility in numerical workflows.

As an example consider the following simple case:

```{code-cell} python
from spycci.core.decorators import protect

@protect
def list_of_lists():
    return [[1., 2., 3.], [4., 5., 6.]]

value = list_of_lists()

print(value)
```
As can be seen, the output of `list_of_lists()` has been changed to a tuple of tuples that cannot be changed.