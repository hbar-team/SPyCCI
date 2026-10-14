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

(Examples-IR-toluene)=
# Computing the infrared spectrum of Toluene

***This example is inspired by section 5.18.2 section of the ORCA 6.1.0 manual.***

In this example, the infrared (IR) spectrum of toluene is computed using ORCA as the computational engine. In addition to fundamental transitions, overtones and combination bands are included through the NEARIR keyword.

The `NEARIR` approach in ORCA provides an approximate treatment of anharmonic effects by combining a harmonic frequency calculation with a semiempirical description of intensities. In particular, the `GFN2-xTB` method is used to estimate anharmonic contributions to transition intensities, while the vibrational frequencies themselves remain essentially harmonic (i.e., they are not explicitly corrected for anharmonicity).

First, a `System` object representing the toluene molecule is created from its SMILES string (`Cc1ccccc1`). An instance of the `OrcaInput` class is then initialized, specifying the `RI-B2PLYP` method together with the `def2-TZVP` basis set.

A geometry optimization is performed using tight convergence criteria (`TightOpt`). This step is particularly important: as noted in the ORCA manual, the quality of anharmonic intensity estimates is highly sensitive to the underlying molecular geometry. Therefore, a well-converged structure is essential, and at least `TightOpt` (or stricter) is recommended.

The frequency calculation is then carried out as a separate step. In this case, numerical differentiation is used to compute vibrational frequencies, and the `overtones=True` option is enabled. This activates the `NEARIR` keyword in the ORCA input, allowing the calculation of overtones and combination bands within the approximate anharmonic framework described above.

The results of the calculations, stored in the `System` object, can then be exported in JSON format for subsequent analysis.

```python
from spycci.systems import System
from spycci.engines.orca import OrcaInput

import logging
logging.basicConfig(level=logging.INFO)

system = System.from_smiles("toluene", "Cc1ccccc1")
orca = OrcaInput(method="RI-B2PLYP", basis_set="def2-TZVP", aux_basis="def2-TZVP/C")

orca.opt(system, optimization_level="TightOpt", frequency_analysis=False, inplace=True)
orca.freq(system, numerical=True, overtones=True, inplace=True)

system.save_json("toluene.json")
```

To compare the obtained results to the experimentally recorded spectrum, an anharmonic correction factor $\lambda_\mathrm{fund}$ can be used to adjust the numerically computed harmonic frequencies to the fundamental ones. For the Weigend−Ahlrichs basis set these factors have been determined and for a RI-B2PLYP functional and def2-TZVP basis set the proper correction factor appears to be 0.9623.[^1]

The experimental spectrum can be obtained from the [NIST webook](https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Mask=80#IR-Spec) in `.jdx` format. 

Starting from the saved JSON file the following script can be used for analysis:

```{code-cell}python
import numpy as np
import matplotlib.pyplot as plt
from spycci.systems import System

# Parse the NIST spectrum to load wavenumbers and transmittance datapoints
frequency, transmittance = [], []
with open("./data/NIST_toluene.jdx", "r") as file:

    deltaf = None
    line = ""
    while "XYDATA" not in line:

        line = file.readline()

        if line.startswith("##DELTAX"):
            sline = line.split("=")
            deltaf = float(sline[1])

    for line in file:

        if "END" in line:
            break

        data = [float(x) for x in line.split()]

        f0 = data[0]
        for i, x in enumerate(data[1::]):
            frequency.append(f0 + i*deltaf)
            transmittance.append(x)

transmittance = np.array(transmittance)

# Define frequency range to plot the spectrum
frange = [min(frequency), max(frequency)]

# Convert transmittance to absorbance for the NIST spectrum
absorbance = -np.log10(transmittance)

# Create matplotlib figure and plot the data from NIST
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(frequency, absorbance, c="#DD0000", label="Experimental (NIST)")

# Load the system data from the `.json` file
system = System.from_json("./data/toluene.json")

# Extract vibrational data from the loaded system and 
# scale by anharmonic correction factor from the literature
vibdata = system.properties.vibrational_data
vibdata.frequencies = [0.9623*f for f in vibdata.frequencies]

# Plot on the same axes the computed spectrum
vibdata.show_ir_spectrum(lineshape="gaussian", axes=ax, range=frange, label="RI-B2PLYP def2-TZVP")

plt.legend()
plt.tight_layout()
plt.show()
```


[^1]: Kesharwani, Manoj K.; Brauer, Brina; Martin, Jan M. L. "Frequency and Zero-Point Vibrational Energy Scale Factors for Double-Hybrid Density Functionals (and Other Selected Methods): Can Anharmonic Force Fields Be Avoided?" *J. Phys. Chem. A*, **2015**, 119 (9), 1701–1714. DOI: [10.1021/jp508422u](https://pubs.acs.org/doi/10.1021/jp508422u).