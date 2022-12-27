# API Documentation

The API documentation represents an object-level description of the package content.
This API reference is divided into the following submodules, matching the structure of the library:

* [`compechem.core`](API-core): all the basic building blocks of the library.

* [`compechem.systems`](API-systems): classes for storing information about the molecular system under study, such as geometry, number of atoms, calculated energies, charge, spin, properties, etc.

* [`compechem.wrappers` and `compechem.engines`](API-wrappers): all the wrappers and engines for interfacing with external software and running calculations, such as **[Orca](https://sites.google.com/site/orcainputlibrary/home)**, **[DFTB+](https://dftbplus.org/)**, **[CREST](https://xtb-docs.readthedocs.io/en/latest/crest.html)**, etc.

* [`compechem.functions`](API-functions): mid-level algorithms for manually calculating specific physical observables, such as pKa, redox potentials, Fukui functions, etc. The user must provide all the necessary species for the calculation, for example the specific protonated and deprotonated species in a pKa calculation.

* [`compechem.tools`](API-tools): various useful functions called internally throughout the code. The user should generally not need to use this module during normal operation.
