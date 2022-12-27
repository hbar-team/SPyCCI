(release-notes)=
# Release notes

## Version `0.3.0`

* Refactoring of `System`:
  * The geometry of the molecule has been moved to a `MolecularGeometry` class that is now an attribute of the `System` class.
  * The properties are now organized in a `Properties` class holding the reference to the electronic and vibronic levels of theory.
  * Added option to save and load `Sysyem` data in `.json` data format.

* Differentiated engines from wrappers
  * Engines: take as argument a `System` object and update its properties
  * Wrappers: general wrappers around a software
  
* Implemented early version of dependency search/check for wrappers and engines
* Implemented VMD based tools to render Fukui functions
* Implemented new parser for orca + added Hirshfeld charges
* Moved constants to a separate file
* Removed `MDTrajectory` class and `velocities` (will be re-implemented with NAMD engine)
* Added some unit and integration tests