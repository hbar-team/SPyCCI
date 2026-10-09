from spycci.systems import System

TEST_DIR = "/home/ppravatto/Documenti/Esperimenti/ChimicaComputazionale/SPyCCI/tests/unit"

mol = System.from_json(f"{TEST_DIR}/utils/json_examples/CO2_ir_raman.json")
obj = mol.properties.vibrational_data


obj.show_raman_spectrum(
    lineshape="gaussian",
    #range=(1000, 2000),
    show_bars=True,
    #figsize=(8, 4),
    color="red",
    #export_path=f"{TEST_DIR}/IR_spectrum_test.png",
    export_dpi=400,
    show=True,
)

