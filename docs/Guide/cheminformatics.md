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

(Guide-cheminformatics)=
# Cheminformatics tools

Until now, the central subject of this guide has been the `System` class that, as has been shown, represents the key object in the definiton of a molecular system and its properties deriving from computational chemistry calculations. The `System` class, in a sense, represents a container of atomic coordinates and derived quantum mechanical properties giving a physical representation of the molecular behavior. What the system class lacks, however, is the concept of valence, connectivity, bond and functional groups that often derives from empirical rules and chemical intuition. Of course these quantities have a quantum mechanical root and can be traced back to quantum mechanical bond orderd. However, in inferring connectivity, these information are often not available and the bond positions and their order needs to be inferred heuristically. This is typical of cheminformatics, where molecules can be represented as a combination of atoms and bonds of different types.

To extend the use of the `System` class to the field of cheminformatics, the `ChemInfo` library has been defined. The `ChemInfo` library wraps a `System` object extending its use to the field of cheminformatics and exposing to the user a broad set of tools to explore molecular connectivity, structural properties and cheminformatic descriptors based on heuristic rules or data not derived from computational chemistry calculations. The core of the class is based on the [RDKit library](https://www.rdkit.org/docs/index.html#) that is tasked with the connectivity determination. The job of inferring connectivity from molecular coordinates is, however, not trivial especially considering the universal nature of the `System` class. This is especially true in the case of non-standard systems i.e. open-shell systems, system displaying non-standard valences or molecule containing coordinated metal atoms. To handle these edge cases an heuristic procedure has been defined within SPyCCI (in the `spycci.tool.rdkittools` module) to extend the use of RDKit to more complex molecular systems. More details about the adopted algorithm can be found in the ["How does the connectivity determination works?" section](Guide-rdkittools).



---

(Guide-rdkittools)=
## How does the connectivity determination works?

The `spycci.tool.rdkittools` module defines an heuristic procedure dedicated to the conversion of a `System` object to a `rdkit.Chem.Mol` one. At its core, the converson is performed by the by the `rdkit.Chem.rdDetermineBonds.DetermineBonds` function of the RDKit package that is tasked with the determination of the molecular connectivity from a set of atomic symbols and three-dimensional cartesian coordinates. While the procedure is straightforward for closed shell system containing main group elements in standard valences and connected by covalent bonds, it often fails in more complex molecular systems requiring chemical intuition to both obtain chemically sensible outputs and avoid exceptions.

To this end, a general heuristic algorithm has been defined. The algorithm starts by initializing a `Mol` object from the list of atom symbols and coordinates. In the process, no implicit hydrogen is considered. If the `System` object has Mulliken spin population values stored in the `system.properties.mulliken_spin_populations` variable these can be used to automatically set a number of radical electros per-atomic site, selected to match the spin multiplicity of the system, using the ceiling function on the atomic spin population value. Using Mulliken spin population values to guide connectivity assignent is particulary useful in non-typical systems such as diradical systems in singlet state or system generated from the addition or abstraction of an electron without lost or addition of a bond (e.g. the $\mathrm{C_6H_6^+}$ benezene radical cation).

If the system is in a singlet state, the generated `Mol` object is directly forewarded to the `DetermineBonds` function that, based on the system charge, proceeds to the determination of the most fitting bonding scheme. The resulting connectivity is then checked for charge and spin and returned to the user. If conversion fails an attempt to compute connectivity using a triplet state is made to cover possible diradical systems not catched by the singlet state routine. (This option has been implemented but it is of limited use since diradical systems not set in singlet state are often converted in anion/cation charge pairs by RDKit)

If the system is in a multiplet state, a more complex procedure is applied: a copy of the `Mol` system is created and connectivity is determined using a charge-shift approach. This is required since in the majority of cases the presence of radical sites causes crashes in the RDKit `DetermineBonds` function that, by design, expects a different charge value for a given molecular system (e.g. the methyl radical cannot be distinguished from the methyl cation by the `DetermineBonds` function, which raises an error). To avoid this problem a charge-shifting approach is applied in which connectivity determination is run with a `try`/`catch` scheme on different charge values. The procedure starts with a neutral system guess (to take into account radical systems generated by addition or removal of an electron) and, in case of failure, runs the connectivity determination using a charge value shifted up or down by a number of electrons theoretically sufficient to bring the system to a singlet state. This gives RDKit a way of processing the system connectivity without crashes, fictitiously changing radical sites into charged ones (e.g. the methyl radical can be processed as a methyl carbocation). Once the guess connectivity generated from the hypothetical singlet system is obtained, it is copied to the original `Mol` object.

Doing so can create instances of a valence mismatch due to the connectivity being copied to a molecular setup with a different electron count. This is the case of many charged main block elements that once processed using a positive or negative charge can result in non-standard valence values (i.e. the nitric oxide molecule $\mathrm{NO_2}$, a neutral radical, is internally processed as a $\mathrm{NO2^+}$ cation that, after the connectivity transfer operation, generates a temporary structure having a nitrogen atom with valence 4 that would make an RDKit sanitization operation fail). To address this issue, an internal valence adjustment routine has been defined: the algorithm is capable of detecting atoms exceeding their maximum valence and fixing their connectivity and valence by breaking multiple bonds and converting them to charge pairs. (i.e. the $\ce{O=N=O}$ structure, having a nitrogen with valence 4, would be converted to a more chemically sensible $\ce{O=N^+=O^-}$ structure ready for sanitation). This approach is automatically applied during the latter part of the workflow and, in what follows, will be referred to as "valence adjustment".

From this point on two paths open: if no radical information has been assigned using Mulliken spin populations, the obtained system is directly passed to the valence adjustment angorithm the output of which is sanitized using the `SANITIZE_PROPERTIES` and `SANITIZE_FINDRADICALS` flags and returned to the user after charge and spin checks; if radical information has been assigned using Mulliken spin populations a more complex approach is followed. First, a copy of the `Mol` object is subjected to valence adjustment and sanitized using only the `SANITIZE_PROPERTIES` flag; if formal charge assignment is sufficient to produce the correct system spin and charge, the result is copied back to the original `Mol` object and returned to the user. If `SANITIZE_PROPERTIES` is not sufficient to produce the correct charge and spin values (i.e. radicals are missing, not compatible with connectivity, or proper formal charges cannot be assigned) a new copy of the `Mol` object is created and sanitized using the `SANITIZE_PROPERTIES` and `SANITIZE_FINDRADICALS` flags catching errors eventually deriving from valence mismatch.

After the first trial sanitization, the radical sites generated from Mulliken spin populations are checked to verify whether they have been altered by the sanitization procedure, indicating an invalid bonding scheme (the guess connectivity generated by the charge-shift scheme is incompatible with a radical site). The affected radical sites are then subjected to a connectivity adjustment procedure that, by altering the bond orders around the affected radical site, tries to generate a sensible connectivity for the radical center. (This is the case of the previously discussed $\mathrm{C_6H_6^+}$ benzene radical cation in which the guess connectivity generated by the charge-shifting procedure is equivalent to that of benzene. As such, connectivity around the radical site must be adjusted by kekulizing the aromatic ring, breaking the double bond connecting the radical site to the adjacent carbon atom, and setting a positive formal charge on the latter). The modified connectivity is then passed to the valence adjustment routine and applied to the `Mol` object, the obtained system is sanitized using the `SANITIZE_PROPERTIES` and `SANITIZE_FINDRADICALS` flags, checked for charge and spin and finally returned to the user.

If the starting system contains metal atoms or ions, the connectivity determination procedure is slightly modified. Metal atoms, especially those in the "d" and "f" blocks, often exhibit complex and variable bonding schemes that are not well-handled by RDKit's `DetermineBonds` function. In particular, the standard valence rules and bond-order heuristics implemented in RDKit are optimized for main-group organic chemistry and typically fail when applied to metal centers or highly ionic species. As a result, attempting to assign connectivity directly in the presence of metals can lead to errors, incorrect bond assignments, or failed sanitization. To overcome this limitation, metal atoms are temporarily removed from the system prior to connectivity assignment. The remaining "organic" backbone (ligand portion) is processed using the previously described workflow by adjusting the charge of the sub-system to account for the removed metals. Once the connectivity of the ligand is established, the metal atoms are reinserted into the RDKit `Mol` object. The last operation is carried out making sure that the final RDKit `Mol` preserves the correct atom ordering and three-dimensional coordinates of the original `System`. 

In the current implementation, user assistance is required to assign the formal charge of the metal ions based on their oxidation state. These must be provided by the user in the form of a dictionary mapping metal atom indices to oxidation states. If no oxidation state is provided the whole system is directly processed using the standard algorithm without any modification.

This approach provides a practical solution to the challenge of including metals in cheminformatics workflows: by decoupling the organic and metallic portions of the system, connectivity can be determined reliably for the ligand, while metal atoms are incorporated in a controlled manner that preserves the integrity of the overall molecular structure.

The overall procedure can be summarized in the following flow chart:

```{graphviz}
digraph {
	node [fillcolor="#AAFFAA" shape=octagon style=filled]
	Start [label="Input `System`"]
	End [label="Check output and `return`"]
	node [fillcolor=white shape=rect style=filled]
	StripMetals [label="Strip metals and run connectivity
on organic part only (ligand)"]
	AddMetals [label="Add metals back as
non-bonded ions"]
	DetermineBonds [label="Apply `DetermineBonds`"]
	ConvertCarbene [label="Convert carbene to singlet
(no radical electrons)"]
	ConvertTriplet [label="Convert input `System` to triplet"]
	GuessConnectivity [label="Get guess connectivity by charge shift"]
	node [fillcolor="#FFEEFF" shape=rect style=filled]
	MolWithRadicals [label="Define `rdchem.Mol` object
with radicals assigned"]
	ValenceAdjMullFirst [label="Run valence adjustment"]
	TrySanitizeProps [label="Try sanitize `PROPERTIES`"]
	TrySanitizePropsRadicals [label="Try sanitize `PROPERTIES`
and `FINDRADICALS`"]
	AdjustRadicalConnectivity [label="Adjust connectivity around
affected radicals"]
	ValenceAdjMullFinal [label="Run valence adjustment"]
	SanitizeFinal [label="Sanitize `PROPERTIES`
and `FINDRADICALS`"]
	node [fillcolor="#EEFFFF" shape=rect style=filled]
	MolCoordsOnly [label="Define `rdchem.Mol` object
with coordinates only"]
	ValenceAdjNoMull [label="Run valence adjustment"]
	SanitizeProps [label="Sanitize `PROPERTIES`
and `FINDRADICALS`"]
	node [fillcolor=lightyellow shape=ellipse style=filled]
	HasMetals [label="Does molecule contain metals
and user-provided oxidation states?"]
	HasMulliken [label="Are Mulliken spin
populations available?"]
	IsSinglet [label="Is the system a singlet?"]
	DoubleRadicals [label="Two radical electrons
assigned to same atom?"]
	ConversionSuccess [label="Was the conversion successful?"]
	RadicalsAssigned [label="Radicals have been set?"]
	ChargeSpinCorrect [label="Are `charge` and `spin` correct?"]
	RadicalsAffected [label="Are set radicals affected or
did sanitation raised valence errors?"]
	Start -> HasMetals
	HasMetals -> HasMulliken [label=NO]
	HasMetals -> StripMetals [label=YES]
	StripMetals -> Start [label="Run connectivity
on ligand only" color=darkgreen dir=both fontcolor=darkgreen style=dashed]
	StripMetals -> AddMetals
	AddMetals -> End
	HasMulliken -> MolCoordsOnly [label=NO]
	HasMulliken -> MolWithRadicals [label=YES]
	MolCoordsOnly -> IsSinglet
	MolWithRadicals -> IsSinglet
	IsSinglet -> DetermineBonds [label=YES]
	DetermineBonds -> DoubleRadicals
	DoubleRadicals -> ConversionSuccess [label=NO]
	DoubleRadicals -> ConvertCarbene [label=YES]
	ConvertCarbene -> ConversionSuccess
	ConversionSuccess -> End [label=YES]
	ConversionSuccess -> ConvertTriplet [label=NO]
	ConvertTriplet -> Start
	IsSinglet -> GuessConnectivity [label=NO]
	GuessConnectivity -> RadicalsAssigned
	RadicalsAssigned -> ValenceAdjNoMull [label=NO]
	ValenceAdjNoMull -> SanitizeProps
	SanitizeProps -> End
	RadicalsAssigned -> ValenceAdjMullFirst [label=YES]
	ValenceAdjMullFirst -> TrySanitizeProps
	TrySanitizeProps -> ChargeSpinCorrect
	ChargeSpinCorrect -> End [label=YES]
	ChargeSpinCorrect -> TrySanitizePropsRadicals [label=NO]
	TrySanitizePropsRadicals -> RadicalsAffected
	RadicalsAffected -> End [label=NO]
	RadicalsAffected -> AdjustRadicalConnectivity [label=YES]
	AdjustRadicalConnectivity -> ValenceAdjMullFinal
	ValenceAdjMullFinal -> SanitizeFinal
	SanitizeFinal -> End
}
```
