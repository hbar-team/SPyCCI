(Guide-run)=
# Running calculations

After creating a `System`, and having initialized a `engines`, we can actually start running calculations.

---

## Generic calculations

Generic calculations are implemented as class methods in the various `engines`. The available options, the class method name, and the engines for which they have been implemented are:

|  Calculation type  |  method name  |  Engines implemented  |
|  :---  |  :---  |  :---  |
|  Single point energy  |  `spe`  |  xTB, Orca, DFTB+  |
|  Geometry optimisation + Frequency analysis  |  `opt`  |  xTB, Orca, DFTB+  |
|  Frequency analysis  |  `freq` |  xTB, Orca  |
|  Numerical frequency analysis  |  `nfreq` |  Orca  |
|  Relaxed surface scan  |  `scan` |  Orca  |
|  NVT Molecular Dynamics  |  `md_nvt` |  DFTB+  |
|  Simulated annealing  |  `simulated_annealing` |  DFTB+  |

---

The only mandatory parameter to be passed is the `System` on which to run the calculation. To run the calculation, simply call the function. The output of a calculation is a `System` object which can be assigned to a new variable:

```python
from compechem.systems import System
from compechem.engines.xtb import XtbInput

mol = System("mymol.xyz")
xtb = XtbInput()

mol_opt = xtb.opt(mol)

print("mol:")
print(mol)
print("mol_opt:")
print(mol_opt)
```

```
mol:
=== System: water ===

Number of atoms: 3
Charge: 0
Spin: 1

--- Warnings ---

--- Energies (Eh) ---

--- Coordinates (Å) ---

O       0.000   -0.736  0.000
H       1.442   0.368   0.000
H       -1.442  0.368   0.000

--- Velocities (Å/ps) ---
```

```
mol_opt:
=== System: water ===

Number of atoms: 3
Charge: 0
Spin: 1

--- Warnings ---

--- Energies (Eh) ---

* Method: gfn2
Electronic: -5.085021284485 Eh
Vibronic: 0.002047859242 Eh

--- Coordinates (Å) ---

O       -0.00000460644656       -0.38108317130711       0.00000066512472
H       0.77437021701733        0.19053853060686        -0.00000303561805
H       -0.77436561057077       0.19054464070025        0.00000237049333

--- Velocities (Å/ps) ---

```

Alternatively, you can directly update the input `System` with the `inplace` flag:

```python
from compechem.systems import System
from compechem.engines.xtb import XtbInput

mol = System("mymol.xyz")
xtb = XtbInput()

print("mol before opt:")
print(mol)

xtb.opt(mol, inplace=True)

print("mol after opt:")
print(mol)
```

```
mol before opt:
=== System: water ===

Number of atoms: 3
Charge: 0
Spin: 1

--- Warnings ---

--- Energies (Eh) ---

--- Coordinates (Å) ---

O       0.000   -0.736  0.000
H       1.442   0.368   0.000
H       -1.442  0.368   0.000

--- Velocities (Å/ps) ---
```

```
mol after opt:
=== System: water ===

Number of atoms: 3
Charge: 0
Spin: 1

--- Warnings ---

--- Energies (Eh) ---

* Method: gfn2
Electronic: -5.085021284485 Eh
Vibronic: 0.002047859242 Eh

--- Coordinates (Å) ---

O       -0.00000460644656       -0.38108317130711       0.00000066512472
H       0.77437021701733        0.19053853060686        -0.00000303561805
H       -0.77436561057077       0.19054464070025        0.00000237049333

--- Velocities (Å/ps) ---

```

---

## Special calculations

Some calculations are instead implemented as functions within the corresponding submodule, and work a little differently from the previous methods.

### CREST

|  Calculation type  |  method name  | 
|  :---  |  :---  | 
|  Tautomer search  |  `tautomer_search`  | 
|  Conformer search  |  `conformer_search`  |
|  Deprotomer search  |  `deprotonate` |
|  Protomer search  |  `protonate` | 
|  Quantum Cluster Growth  |  `qcg_grow` |
|  Quantum Cluster Growth + ensemble evaluation  |  `qcg_ensemble` |


These can be imported directly from the corresponding submodule. For more details on each specific function, please refer to the corresponding [API](API-wrappers-crest) page.

The CREST tautomer, conformer, deprotonation, and protonation routines all require a `System` object as input, and return a list of tautomer, conformers, deprotomers, and protomers, respectively, ordered by relative energy:

```python
from compechem.systems import System
from compechem.wrappers.crest import conformer_search

mol = System("mymol.xyz")
conformers = conformer_search(mol)

lowest_energy_conformer = conformers[0]
```

The QCG routines are used to create an explicitly solvated cluster, surrounding a solute with a certain number of solvent molecules. By default, the program keeps adding solvent molecules until energy convergence is reached with respect to the addition of other molecules. For more details on how to customise this kind of calculation, please refer to the [API](API-wrappers-crest) page.

```python
from compechem.systems import System
from compechem.wrappers.crest import qcg_grow

solute = System("solute.xyz")
solvent = System("solvent.xyz")

cluster = qcg_grow(solute, solvent)
```

### Packmol

The Packmol wrapper allows for the construction of explicitly solvated systems, as a fast alternative to the QCG routines, useful for setting up MD calculations. The currently available functions are:

|  Calculation type  |  method name  | 
|  :---  |  :---  | 
|  Generation of explicit solvation cube  |  `packmol_cube`  | 

Like the QCG routines, you need to provide a solute and a solvent `System`. The `packmol_cube` routine however allows you a finer degree of control, as you also need to provide two of the following three parameters:

  * `nsolv`: number of solvent molecules to put in the cube
  * `target_dens`: target density of the solvated box, in g/L
  * `cube_side`: length of the simulation box, in Å

Given two of these parameters, the third will be calculated accordingly. For example, the following code generates a cubic solvation box with 50 solvent molecules and a target density of 997 g/L:

```python
from compechem.systems import System
from compechem.wrappers.packmol import packmol_cube

solute = System("solute.xyz")
solvent = System("solvent.xyz")

solvated_cube = packmol_cube(solute, solvent, nsolv=50, target_dens=997)
print(solvated_cube)
```

```
=== System: urea_50waters ===

Number of atoms: 158
Charge: 0
Spin: 1

--- Warnings ---

--- Energies (Eh) ---

--- Coordinates (Å) ---

C       5.67500 5.59400 6.27900
N       5.71300 4.95300 5.03000
H       5.90100 3.95900 5.06700
H       6.17200 5.43600 4.26800
N       5.71500 6.99700 6.23800
H       6.16900 7.43100 5.44400
H       5.90800 7.44500 7.12600
O       5.53500 4.97100 7.33300
H       2.90600 11.30900        9.82900
H       3.10100 9.69600 9.66300
O       3.57300 10.57700        9.69100
H       10.81500        5.61800 8.60100
H       11.74700        6.62100 9.49100
O       11.00700        5.95000 9.52500
H       1.42900 11.26600        11.46900
H       2.64700 10.20900        11.72600
O       1.71500 10.31100        11.38000
H       9.39700 7.79600 7.73200
H       10.77400        8.65600 7.90700
O       10.36700        7.74500 7.97000
H       10.35700        4.25900 11.32800
H       10.56700        5.84000 11.67800
O       11.03400        4.97900 11.47800
H       11.39800        5.98000 2.11400
H       10.24000        4.86600 1.82300
O       11.22400        5.04000 1.82000
H       10.46800        1.13000 1.32300
H       11.30100        2.52300 1.13800
O       11.38000        1.53200 1.24300
H       4.91200 3.04200 9.77300
H       4.66100 3.14100 8.16200
O       4.55400 2.56400 8.97000
H       9.47800 10.32500        11.69900
H       8.18200 9.34900 11.51300
O       8.48400 10.29800        11.59400
H       2.81600 0.99900 9.18700
H       1.98200 2.39200 9.35400
O       1.90300 1.40500 9.22300
H       6.93400 4.54800 9.99800
H       7.36000 4.39900 8.42900
O       6.64000 4.22300 9.09900
H       1.89500 10.77200        3.13000
H       3.27000 11.57200        3.49600
O       2.28800 11.51700        3.66900
H       9.33800 6.21400 5.46100
H       9.73100 5.45700 4.06800
O       8.97500 5.79700 4.62700
H       1.56300 8.04600 7.83300
H       2.00000 6.66100 7.08700
O       2.33500 7.51500 7.48500
H       5.45200 11.69800        10.95200
H       5.96500 10.25200        11.50900
O       6.11000 10.95700        10.81600
H       10.21800        11.15500        6.90600
H       11.61000        11.37000        7.73100
O       10.63400        11.17400        7.81500
H       4.59200 4.84200 10.95600
H       5.30000 6.11100 11.70200
O       4.43400 5.73900 11.37000
H       4.52600 6.39200 8.77900
H       4.17900 7.42900 9.99300
O       3.84600 7.03200 9.13800
H       10.33400        0.99800 5.04900
H       9.41900 2.35000 5.11300
O       9.41800 1.35700 5.23200
H       10.37500        8.36500 5.65100
H       9.27900 9.56900 5.78200
O       10.24800        9.35800 5.65900
H       9.95100 4.03700 6.52300
H       8.53000 3.33500 6.91500
O       8.97600 4.19800 6.68000
H       11.64100        9.78700 3.15500
H       10.08300        10.23500        2.95900
O       10.69900        9.45800 3.07900
H       5.27600 11.70000        6.51800
H       6.45600 10.57600        6.42200
O       5.48300 10.75200        6.27700
H       10.47900        11.73000        4.50300
H       11.72700        11.69200        5.55500
O       11.37700        11.33200        4.69100
H       3.99200 9.05800 4.66900
H       4.96500 10.27500        4.18000
O       4.49500 9.44100 3.89400
H       0.99800 11.67200        5.89500
H       0.99700 10.04100        5.97000
O       1.45700 10.84000        5.58400
H       3.33600 11.70000        7.60900
H       2.87500 10.13700        7.50200
O       3.15200 10.94300        6.98100
H       8.16600 2.80400 11.71900
H       8.93800 1.38600 11.46900
O       9.05600 2.37400 11.56400
H       9.33800 9.39800 9.67400
H       10.33300        8.18300 10.12300
O       9.39000 8.42400 9.89400
H       10.11400        9.35700 0.97700
H       11.74500        9.41900 1.00200
O       10.90800        9.96400 0.97000
H       3.59600 2.04400 4.52500
H       3.49200 3.64900 4.24400
O       4.05500 2.83300 4.11800
H       2.34300 1.32100 7.04000
H       2.66900 2.89400 7.33100
O       3.06600 1.98200 7.24200
H       8.22400 5.88300 11.60400
H       8.96400 7.33600 11.69100
O       8.08200 6.87300 11.61700
H       11.30900        3.51000 8.25900
H       11.07600        2.69700 9.65600
O       11.72300        3.25000 9.13100
H       3.07700 3.09300 11.21500
H       1.59100 3.76700 11.15500
O       2.57200 3.95300 11.13700
H       11.67400        6.81400 6.52000
H       11.70500        5.19700 6.74500
O       11.41100        5.93000 6.13200
H       8.47400 10.22100        7.84500
H       7.71100 11.64500        8.08700
O       8.26100 10.93000        8.51800
H       9.99200 1.35100 7.46700
H       11.36300        1.89300 6.76600
O       10.99100        1.35000 7.51800
H       2.01100 7.81400 10.23800
H       1.00000 6.60300 9.81700
O       1.21600 7.27700 10.52200
H       8.58700 8.78200 3.56700
H       9.68300 7.57200 3.56700
O       8.70800 7.78900 3.56700
H       9.41200 4.03300 9.18800
H       8.96600 2.46800 9.04700
O       8.84600 3.30600 9.57800
H       10.55000        11.59300        10.27800
H       11.33800        10.16200        10.23800
O       11.44900        11.15600        10.22900
H       1.52100 7.58200 3.27100
H       2.99000 7.66000 2.56100
O       2.21900 8.19700 2.90200
H       6.95000 1.94500 10.20500
H       5.81800 0.92900 9.61100
O       6.79900 1.08100 9.72400
H       1.59400 3.81200 1.92200
H       1.06900 2.30200 1.58700
O       1.87000 2.85700 1.80700
H       6.44500 9.32500 9.05900
H       7.21500 8.06400 9.75400
O       6.75200 8.93200 9.92700
H       9.84900 2.77700 2.99400
H       11.38900        2.92200 3.51700
O       10.50700        3.38500 3.44000
H       4.94300 1.99600 11.67600
H       3.65200 1.00100 11.58100
O       4.64900 1.04100 11.62600
H       0.99900 2.00800 4.74700
H       0.99900 3.62500 4.97600
O       1.57400 2.81000 4.91000
H       6.27500 9.33000 2.04200
H       5.19600 10.20100        1.18100
O       5.68800 10.13900        2.04900

--- Velocities (Å/ps) ---

```