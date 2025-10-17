(changelog)=
# Release notes

## Version `1.0.0`

* **Full integration of the `GES-comp-echem` library (v0.3.2)** and complete code architecture refactoring.  
  Main breaking changes:
  - Full revision of the `__init__` scheme of the `System` class, with the introduction of dedicated `@classmethod`s for parsing different file types.  
  - Revised policy on data origin strictness: the `properties` argument of the `System` class is now protected against unconditional mixing of data obtained from calculations performed at different levels of theory.

* Implemented new dependency finder protocol to improve stability and ensure scalability.

* Implemented generation of `System` objects from **SMILES** strings.

* Implemented support for solvation energy calculations based on **OpenCOSMO-RS** (for `orca >= 6.0.0`).

* Introduced new protocols for **pKa computation** based on the oxonium scheme, including COSMO-RS corrections.

* Upgraded and extended support for **`vmd`-based visualization tools** to assist users in visualizing molecular systems and cube files. Improvements to the tools dedicated to cube file parsing and manipulation.

* Updated and expanded documentation, providing more in-depth explanations of library mechanics and the approximations involved.

* Developed and updated test suite**, and **implemented continuous integration (CI)** for test execution using Docker.
