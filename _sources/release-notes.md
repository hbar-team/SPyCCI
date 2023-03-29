(changelog)=
# Release notes

## Version `0.3.1`

* Update of `System` and `JSON` format:
  * Defined extension-based identification of `.xyz` and `.json` file types in `System` class constructor.
  * Defined versioning of `JSON` files and auto-update routines to ensure retro-compatibility

* Update to the engines classes behavior and testing:
  * Charge and spin cannot be changed from engine during calculation (Must be changed manually by the user acting on `System`)
  * Added test for non-inplace calculations after bugfix.
  * Added explicit option in `OrcaInput` to run frequency analysis when running optimization. Removed automatic switch to numerical frequencies when requiring analytical frequencies in solvent.
  * Added option to select the level of geometry convergence to be used during optimization in `OrcaInput`.

* Added vibrational spectroscopy functions to `OrcaInput` parser:
  * Added `VibrationalData` object in properties to store vibrational informations and plot infrared and raman spectra.
  * Added option to compute Raman spectra when calling frequency analysis.
  * Added option to compute overtones and combination bands in infrared spectra when calling frequency analysis.

* Implemented buried volume calculation from `System` geometry.

* Added `mogli` tools to visualize molecule and use custom coloring (early)



## Version `0.3.0`

* Updated `functions` and added automatic routine for computing pKa
  * Added functional tests
  * Updated documentation

* `Properties` class now accepts also levels of theory in string format and performs an early validation of the format

* Added gibbs free energy parser to `OrcaInput` and `XtbInput` classes

* Moved `MPI_FLAGS` to the `config` module as global variables

* Various bugfixes and code uniformation
  * VMD based wrapper no longer strips characters at the beginning of filenames
  * Mulliken charges and spin populations are correctly parsed for coupled-cluster calculations
  * Removed forbidden characters to output files (e.g. partentheses and slash)

## Version `0.3.0 - alpha`

* Refactoring of `System`:
  * The geometry of the molecule has been moved to a `MolecularGeometry` class that is now an attribute of the `System` class.
  * The properties are now organized in a `Properties` class holding the reference to the electronic and vibronic levels of theory.
  * Added option to save and load `System` data in `.json` data format.

* Differentiated engines from wrappers
  * Engines: take as argument a `System` object and update its properties
  * Wrappers: general wrappers around a software
  
* Implemented early version of dependency search/check for wrappers and engines
* Implemented VMD based tools to render Fukui functions
* Implemented new parser for orca + added Hirshfeld charges
* Moved constants to a separate file
* Removed `MDTrajectory` class and `velocities` (will be re-implemented with NAMD engine)
* Added some unit and integration tests