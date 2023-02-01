(Guide-functions)=
# Analysing results

The `compechem.functions` submodule contains a series of methods to "manually" calculate some physical properties of the system under study. The user is expected to provide the exact intended states for the system(s) under study, as the functions will simply take the inputs and generate the output without carrying out any checks. For example, if the user calculates the pKa of a molecule in an unoptimised geometry, the resulting pKa will be that of the unoptimised geometry.

The `compechem.functions` functions can be imported via the following syntax:

```python
from compechem.functions import calculate_pka
```

---

## pKa

The `compechem.functions.calculate_pka` function calculates the pKa of a molecule $HA$ considering a reaction of the type:

$$
HA \rightarrow H^{+} + A^{-}
$$

provided the following arguments:

* `protonated` (`System`): molecule in its protonated form
* `deprotonated` (`System`): molecule in its deprotonated form
* `method_el` (`System`): level of theory for the electronic component of the total energy (must be present in the `System.energies` dictionary)
* `method_vib` (`System`): level of theory for the vibronic component of the total energy (must be present in the `System.energies` dictionary). If not specified, defaults to the same level of theory as the electronic energy

The function returns the pKa of the molecule considering the provided states, calculated as:

$$
pK_{a} = \frac{G_{A^{-}} + G_{H^{+}} - G_{HA}}{2.303 \cdot RT}
$$

Where $G_{A^{-}}$ and $G_{HA}$ are calculated summing the electronic + vibronic energies at the selected level of theory, $G_{H^{+}} = -270.29 kcal/mol$, $R = 1.987 \cdot 10^{-3} kcal/(mol \cdot K)$ and $T = 298.15 K$

---

## 1-el redox potential

Calculates the one-electron reduction potential of a molecule $MH_{n}$, considering a generic reaction of the type:

$$
MH_{n} \rightarrow M^{\cdot (n-1)-} + n\cdot H^{+} + e^{-}
$$

provided the following arguments:

* `oxidised` (`System`): molecule in its oxidised state
* `reduced` (`System`): molecule in its reduced state
* `method_el` (`System`): level of theory for the electronic component of the total energy (must be present in the `System.energies` dictionary)
* `method_vib` (`System`): level of theory for the vibronic component of the total energy (must be present in the `System.energies` dictionary). If not specified, defaults to the same level of theory as the electronic energy
* `pH` (`float`, default: `7.0`): pH at which the reduction potential is calculated

and returns the reduction potential of the molecule considering the provided states at the provided pH, including eventual PCET mechanisms, calculated as:

$$
E°_{MH_{n}/M^{\cdot(n-1)-}} = - \frac{G_{M^{\cdot(n-1)-}} - (G_{MH_{n}} + n \cdot G_{H^{+}}) }{F} - E°_{SHE} - n \cdot 0.059 \cdot pH 
$$

where $G_{M^{\cdot(n-1)-}}$ and $G_{MH_{n}}$ are calculated summing the electronic + vibronic energies at the selected level of theory, $G_{H^{+}} = -270.29 kcal/mol$, $F = 23.061 kcal/volt–gram-equivalent$, and $E_{SHE} = 4.28 V$.

## Fukui functions
The `compechem.functions.calculate_fukui` function calculates the Fukui functions $f^+(r)$, $f^-(r)$ and $f^0(r)$ associated with a given molecular geometry. The Fukui functions are computed according to the definitions:

$$
f^+(r) = \rho_{N+1}(r) - \rho_{N}(r)
$$

$$
f^-(r) = \rho_{N}(r) - \rho_{N-1}(r)
$$

$$
f^0(r) = \frac{1}{2} \left[\rho_{N+1}(r) - \rho_{N-1}(r)\right]
$$

Where, given a molecule with $N$ electrons, $\rho_{N}(r)$ represents its electronic density while $\rho_{N\pm1}(r)$ represents the electronic density of the molecule, in the same nuclear configuration, when one electron is either added ($+1$) or removed ($-1$).

The Fukui functions are both computed as volumetric quantities and saved in a [Gaussian Cube](http://paulbourke.net/dataformats/cube/) compatible format in the `output_density` folder and as condensed values saved in the `System` object `properties` attribute in the form of a dictionary. The condensed Fukui functions are computed by applying the $f^+$, $f^-$ and $f^0$ definitions replacing the charge density with either the Mulliken charges or the Hirshfeld charges (changing the sign accordingly given that a localized electronic density represents an accumulation of electrons hence of negative charge). Please notice how the Hirshfeld charges are supported only by the `OrcaInput` engine.

::::{important}
Please notice how the Fukui cubes contain the localized Mulliken-charge-based Fukui values in place of the atomic charges. This is explained in the first comment line of each cube file and, for sake of clarity, all the files are saved using the extension `.fukui.cube`.
::::

The function can be called with the following minimal arguments:
* `molecule` (`System`): The molecular structure to be used in the computation
* `engine` (`OrcaInput` or `XtbInput`): The engine defining the level of theory to be used in the calculation.

The function assumes that the molecule supports only singlet and doublet states and switches the spin multeplicity according to the number of electrons. If different spin states needs to be considered the `spins_states` option can be used to provide the spin multeplicity values as a list.

An example code snippet is provided in what follows:
```python
mol = System("./acetaldehyde.xyz")
orca = OrcaInput(method="M062X", basis_set="def2-TZVP")

orca.opt(mol, inplace=True)
calculate_fukui(mol, orca)

print(mol)
```

That for the acetaldehyde molecule returns the following result:
```
=========================================================
SYSTEM: acetaldehyde
=========================================================

Number of atoms: 7
Charge: 0
Spin multeplicity: 1

********************** GEOMETRY *************************

Total system mass: 44.0526 amu

----------------------------------------------
 index  atom    x (Å)      y (Å)      z (Å)   
----------------------------------------------
 0       C    -3.01613    0.18024    0.12796  
 1       C    -3.66703    1.53166    0.10866  
 2       H    -3.20495    2.18867    0.84147  
 3       H    -4.73073    1.41446    0.32944  
 4       H    -3.59505    1.96256   -0.89180  
 5       O    -2.16724   -0.15157    0.90619  
 6       H    -3.37637   -0.53412   -0.63890  
----------------------------------------------

********************** PROPERTIES *************************

Electronic level of theory: OrcaInput || method: M062X | basis: def2-TZVP | solvent: None
Vibronic level of theory: OrcaInput || method: M062X | basis: def2-TZVP | solvent: None

Electronic energy: -153.820570037973 Eh
Vibronic energy: 0.03120153 Eh
Helmholtz free energy: None Eh
Gibbs free energy: None Eh
pKa: None

MULLIKEN ANALYSIS
----------------------------------------------
 index  atom   charge    spin
----------------------------------------------
 0       C    0.14039   0.00000  
 1       C    -0.38791  0.00000  
 2       H    0.14438   0.00000  
 3       H    0.14899   0.00000  
 4       H    0.14587   0.00000  
 5       O    -0.27129  0.00000  
 6       H    0.07956   0.00000  

CONDENSED FUKUI - MULLIKEN
----------------------------------------------
 index  atom    f+      f-      f0
----------------------------------------------
 0       C    0.43686   0.09908   0.26797  
 1       C    -0.12756  0.01427   -0.05665 
 2       H    0.09539   0.07534   0.08537  
 3       H    0.09967   0.09191   0.09579  
 4       H    0.09621   0.09169   0.09395  
 5       O    0.23951   0.42159   0.33055  
 6       H    0.15992   0.20612   0.18302  

HIRSHFELD ANALYSIS
----------------------------------------------
 index  atom   charge    spin
----------------------------------------------
 0       C    0.15867   0.00000  
 1       C    -0.07114  0.00000  
 2       H    0.04496   0.00000  
 3       H    0.04637   0.00000  
 4       H    0.04447   0.00000  
 5       O    -0.25678  0.00000  
 6       H    0.03345   0.00000  

CONDENSED FUKUI - HIRSHFELD
----------------------------------------------
 index  atom    f+      f-      f0
----------------------------------------------
 0       C    0.31068   0.16128   0.23598  
 1       C    0.07070   0.09203   0.08137  
 2       H    0.05364   0.05358   0.05361  
 3       H    0.08273   0.06865   0.07569  
 4       H    0.07600   0.06840   0.07220  
 5       O    0.27091   0.40759   0.33925  
 6       H    0.13535   0.14846   0.14191
```

The volumetric fukui functions can then be plotted using the built in `vmd` based rendering tool. As an example the following code can be used to render the the $f^+(r)$ Fukui function.

```python
from compechem.tools.vmdtools import render_fukui_cube

render_fukui_cube(
    "./output_densities/acetaldehyde_Fukui_plus.fukui.cube",
    include_negative=True,
    isovalue=0.02,
)
```

The rendered volume is saved in a `.bmp` bitmap image format. In the case of the acetaldehyde molecule considered in the example, the following image is obtained:

```{image} ../images/acetaldehyde_fukui_plus.bmp
:alt: fukui_plus
:class: bg-primary mb-1
:width: 600px
:align: center
```