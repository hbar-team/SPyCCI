from graphviz import Digraph

def create_function_flow() -> Digraph:
    """
    Create a function flow diagram for molecule processing.
    Returns a Graphviz Digraph object.
    """
    dot = Digraph(comment="Function Flow Diagram")
    
    # -----------------------------
    # Node Styles
    # -----------------------------
    start_end_style = {"shape": "octagon", "style": "filled", "fillcolor": "#AAFFAA"}
    op_style = {"shape": "rect", "style": "filled", "fillcolor": "white"}
    op_style_mulliken = {"shape": "rect", "style": "filled", "fillcolor": "#FFEEFF"}
    op_style_nomulliken = {"shape": "rect", "style": "filled", "fillcolor": "#EEFFFF"}
    cond_style = {"shape": "ellipse", "style": "filled", "fillcolor": "lightyellow"}
    
    # -----------------------------
    # Start / End Nodes
    # -----------------------------
    dot.attr("node", **start_end_style)
    dot.node("Start", "Input `System`")
    dot.node("End", "Check output and `return`")
    
    # -----------------------------
    # Operation Nodes
    # -----------------------------
    dot.attr("node", **op_style)
    dot.node("StripMetals", "Strip metals and run connectivity\non organic part only (ligand)")
    dot.node("AddMetals", "Add metals back as\nnon-bonded ions")
        
    dot.node("DetermineBonds", "Apply `DetermineBonds`")
    dot.node("ConvertCarbene", "Convert carbene to singlet\n(no radical electrons)")
    dot.node("ConvertTriplet", "Convert input `System` to triplet")
    
    dot.node("GuessConnectivity", "Get guess connectivity by charge shift")
    
    dot.attr("node", **op_style_mulliken)
    dot.node("MolWithRadicals", "Define `rdchem.Mol` object\nwith radicals assigned")
    dot.node("TrySanitizeProps", "Try sanitize `PROPERTIES`")
    dot.node("TrySanitizePropsRadicals", "Try sanitize `PROPERTIES`\nand `FINDRADICALS`")
    dot.node("AdjustRadicalConnectivity", "Adjust connectivity around\naffected radicals")
    dot.node("SanitizeFinal", "Sanitize `PROPERTIES`\nand `FINDRADICALS`")

    
    dot.attr("node", **op_style_nomulliken)
    dot.node("MolCoordsOnly", "Define `rdchem.Mol` object\nwith coordinates only")
    dot.node("SanitizeProps", "Sanitize `PROPERTIES`\nand `FINDRADICALS`")

    # -----------------------------
    # Conditional Nodes
    # -----------------------------
    dot.attr("node", **cond_style)
    dot.node("HasMetals", "Does molecule contain metals\nand user-provided oxidation states?")
    dot.node("HasMulliken", "Are Mulliken spin\npopulations available?")
    dot.node("IsSinglet", "Is the system a singlet?")
    dot.node("DoubleRadicals", "Two radical electrons\nassigned to same atom?")
    dot.node("ConversionSuccess", "Was the conversion successful?")
    dot.node("RadicalsAssigned", "Radicals have been set?")
    dot.node("ChargeSpinCorrect", "Are `charge` and `spin` correct?")
    dot.node("RadicalsAffected", "Are set radicals affected?")
    
    # -----------------------------
    # Edges
    # -----------------------------
    dot.edge("Start", "HasMetals")
    
    # Metal handling
    dot.edge("HasMetals", "HasMulliken", label="NO")
    dot.edge("HasMetals", "StripMetals", label="YES")
    dot.edge(
        "StripMetals", "Start",
        dir="both",
        label="Run connectivity\non ligand only",
        style="dashed",
        color="darkgreen",
        fontcolor="darkgreen"
    )
    dot.edge("StripMetals", "AddMetals")
    dot.edge("AddMetals", "End")
    
    # Mulliken branch
    dot.edge("HasMulliken", "MolCoordsOnly", label="NO")
    dot.edge("HasMulliken", "MolWithRadicals", label="YES")
    
    dot.edge("MolCoordsOnly", "IsSinglet")
    dot.edge("MolWithRadicals", "IsSinglet")
    
    # Singlet branch
    dot.edge("IsSinglet", "DetermineBonds", label="YES")
    dot.edge("DetermineBonds", "DoubleRadicals")
    dot.edge("DoubleRadicals", "ConversionSuccess", label="NO")
    dot.edge("DoubleRadicals", "ConvertCarbene", label="YES")
    dot.edge("ConvertCarbene", "ConversionSuccess")
    dot.edge("ConversionSuccess", "End", label="YES")
    dot.edge("ConversionSuccess", "ConvertTriplet", label="NO")
    dot.edge("ConvertTriplet", "Start")
    
    # Non-singlet branch
    dot.edge("IsSinglet", "GuessConnectivity", label="NO")
    dot.edge("GuessConnectivity", "RadicalsAssigned")
    dot.edge("RadicalsAssigned", "SanitizeProps", label="NO")
    dot.edge("SanitizeProps", "End")
    dot.edge("RadicalsAssigned", "TrySanitizeProps", label="YES")
    dot.edge("TrySanitizeProps", "ChargeSpinCorrect")
    dot.edge("ChargeSpinCorrect", "End", label="YES")
    dot.edge("ChargeSpinCorrect", "TrySanitizePropsRadicals", label="NO")
    dot.edge("TrySanitizePropsRadicals", "RadicalsAffected")
    dot.edge("RadicalsAffected", "End", label="NO")
    dot.edge("RadicalsAffected", "AdjustRadicalConnectivity", label="YES")
    dot.edge("AdjustRadicalConnectivity", "SanitizeFinal")
    dot.edge("SanitizeFinal", "End")
    
    return dot

# ==============================
# Usage
# ==============================
if __name__ == "__main__":
    graph = create_function_flow()
    graph.render("function_flow", format="svg", view=True, cleanup=True)
    print(graph.source)
