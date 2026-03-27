"""Example: Generate input files for various calculation types.

Demonstrates the Python API for input generation without the agent.
"""

from chem_copilot.generators import FHIAimsGenerator, Psi4Generator
from chem_copilot.models import (
    BasisSetTier,
    CalculationType,
    DispersionMethod,
    FHIAimsParams,
    MoleculeSpec,
    Psi4Method,
    Psi4Params,
    XCFunctional,
    XDMParams,
)

# --- Define molecules ---

water = MoleculeSpec(
    symbols=["O", "H", "H"],
    coords=[[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
)

benzene = MoleculeSpec(
    symbols=["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
    coords=[
        [1.214, 0.700, 0.0], [1.214, -0.700, 0.0], [0.0, -1.399, 0.0],
        [-1.214, -0.700, 0.0], [-1.214, 0.700, 0.0], [0.0, 1.399, 0.0],
        [2.157, 1.245, 0.0], [2.157, -1.245, 0.0], [0.0, -2.490, 0.0],
        [-2.157, -1.245, 0.0], [-2.157, 1.245, 0.0], [0.0, 2.490, 0.0],
    ],
)

silicon = MoleculeSpec(
    symbols=["Si", "Si"],
    coords=[[0.0, 0.0, 0.0], [1.354, 1.354, 1.354]],
    lattice_vectors=[[0.0, 2.708, 2.708], [2.708, 0.0, 2.708], [2.708, 2.708, 0.0]],
)

# --- FHI-aims examples ---

aims = FHIAimsGenerator()

# 1. PBE0 + XDM / tight for a molecule (GMTKN55-style)
print("=" * 60)
print("1. FHI-aims: Benzene PBE0+XDM/tight")
print("=" * 60)
result = aims.generate(benzene, FHIAimsParams(
    xc=XCFunctional.PBE0,
    basis=BasisSetTier.TIGHT,
    dispersion=DispersionMethod.XDM,
    xdm_params=XDMParams(keyword="bj tight"),
))
print(result.summary)
print(result.files["control.in"][:400] + "...\n")

# 2. PBE / tight for Si bulk with k-grid
print("=" * 60)
print("2. FHI-aims: Silicon bulk PBE/tight k=8x8x8")
print("=" * 60)
result = aims.generate(silicon, FHIAimsParams(
    xc=XCFunctional.PBE,
    basis=BasisSetTier.TIGHT,
    k_grid=[8, 8, 8],
))
print(result.summary)
print(result.files["control.in"][:400] + "...\n")

# 3. HSE06 geometry optimization
print("=" * 60)
print("3. FHI-aims: Water HSE06/light optimization")
print("=" * 60)
result = aims.generate(water, FHIAimsParams(
    xc=XCFunctional.HSE06,
    basis=BasisSetTier.LIGHT,
    calculation_type=CalculationType.OPTIMIZATION,
    xc_pre=("pbe", 15),
))
print(result.summary)
print(result.files["control.in"][:500] + "...\n")

# --- Psi4 examples ---

psi4 = Psi4Generator()

# 4. CCSD(T)/cc-pVTZ
print("=" * 60)
print("4. Psi4: Water CCSD(T)/cc-pVTZ")
print("=" * 60)
result = psi4.generate(water, Psi4Params(
    method=Psi4Method.CCSD_T,
    basis="cc-pVTZ",
))
print(result.summary)
print(result.files["input.py"][:400] + "...\n")

# 5. B3LYP-D3BJ/def2-TZVPPD optimization
print("=" * 60)
print("5. Psi4: Benzene B3LYP-D3BJ/def2-TZVPPD optimization")
print("=" * 60)
result = psi4.generate(benzene, Psi4Params(
    method=Psi4Method.DFT,
    functional="b3lyp",
    basis="def2-TZVPPD",
    dispersion=DispersionMethod.D3BJ,
    calculation_type=CalculationType.OPTIMIZATION,
    memory="8 GB",
    num_threads=16,
))
print(result.summary)
print(result.files["input.py"][:400] + "...\n")
