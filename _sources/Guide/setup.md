(Guide-setup)=
# Setting up calculations

After creating a `System`, you need to set up the calculation to which you wish to subject it. This is done with the `compechem.engines` submodule.

The `compechem.engines` submodule contains a series of program-specific submodules for interfacing with external code. The currently supported programs are:

* [xTB](https://github.com/grimme-lab/xtb)
* [Orca](https://orcaforum.kofo.mpg.de/index.php?sid=3c6c78cae3dd0cfffa26a293953422e3)
* [DFTB+](https://dftbplus.org/)
* [NAMD](http://www.ks.uiuc.edu/Research/namd/) (coming soon!)

:::{admonition} Note
:class: warning
To function with the library, the external programs need to be already installed and available to the system from command line!
:::

These general-purpose engines are implemented as `<Program>Input` classes in the corresponding submodules. To initialise a wrapper, you need to import it from its submodule and then create an instance of it:

```python
from compechem.engines.xtb import XtbInput
from compechem.engines.dftbplus import DFTBInput
from compechem.engines.orca import OrcaInput

xtb = XtbInput()
dftb = DFTBInput()
orca = OrcaInput()
```

If you do not specify anything, some default options are chosen automatically for level of theory, basis set, solvation, etc. Please refer to the [API](API-wrappers) section for a complete list of options and default values.

As an example, let us set up a calculation with Orca, using the B3LYP functional, with the def2-TZVP basis set and def2/J auxiliary basis set, using the SMD implicit solvation model for water, and including Grimme's D3BJ dispersion corrections:

```python
from compechem.engines.orca import OrcaInput

b3lyp = OrcaInput(
    method = "B3LYP",
    basis_set = "def2-TZVP",
    aux_basis = "def2/J",
    solvent = "water",
    optionals = "D3BJ",
)
```