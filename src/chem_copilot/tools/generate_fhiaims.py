"""LangGraph tool: FHI-aims input file generation."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from chem_copilot.generators.fhi_aims import FHIAimsGenerator
from chem_copilot.models import (
    BasisSetTier,
    CalculationType,
    DispersionMethod,
    FHIAimsParams,
    MoleculeSpec,
    RelativisticTreatment,
    XCFunctional,
    XDMParams,
)


class FHIAimsInput(BaseModel):
    """Input schema for the FHI-aims generator tool."""

    symbols: list[str] = Field(description="Element symbols, e.g. ['O', 'H', 'H']")
    coords: list[list[float]] = Field(
        description="Cartesian coordinates in Angstrom, shape (N, 3)"
    )
    charge: int = Field(default=0, description="Net charge of the system")
    multiplicity: int = Field(default=1, description="Spin multiplicity (2S+1)")
    lattice_vectors: Optional[list[list[float]]] = Field(
        default=None, description="Lattice vectors for periodic systems, shape (3, 3)"
    )
    xc: str = Field(default="pbe", description="XC functional: pbe, pbe0, hse06, b3lyp, b86bpbe, scan, etc.")
    basis: str = Field(default="light", description="Basis tier: light, intermediate, tight, really_tight")
    calculation_type: str = Field(
        default="single_point",
        description="Calculation type: single_point, optimization, molecular_dynamics, frequency",
    )
    dispersion: str = Field(default="none", description="Dispersion: none, tkatchenko-scheffler, mbd, xdm, etc.")
    xdm_a1: Optional[float] = Field(default=None, description="XDM a1 damping parameter (if dispersion=xdm)")
    xdm_a2: Optional[float] = Field(default=None, description="XDM a2 damping parameter (if dispersion=xdm)")
    k_grid: Optional[list[int]] = Field(default=None, description="k-point grid [k1, k2, k3] for periodic systems")
    compute_forces: bool = Field(default=False, description="Request force computation")
    relativistic: str = Field(default="atomic_zora", description="Relativistic treatment: none, atomic_zora, zora")


@tool("generate_fhiaims_input", args_schema=FHIAimsInput)
def generate_fhiaims_input(
    symbols: list[str],
    coords: list[list[float]],
    charge: int = 0,
    multiplicity: int = 1,
    lattice_vectors: Optional[list[list[float]]] = None,
    xc: str = "pbe",
    basis: str = "light",
    calculation_type: str = "single_point",
    dispersion: str = "none",
    xdm_a1: Optional[float] = None,
    xdm_a2: Optional[float] = None,
    k_grid: Optional[list[int]] = None,
    compute_forces: bool = False,
    relativistic: str = "atomic_zora",
) -> str:
    """Generate FHI-aims control.in and geometry.in input files.

    Given a molecular or periodic system and calculation parameters,
    produces ready-to-run FHI-aims input files. Returns the file contents
    and a summary of the calculation setup.

    Use this tool when the user wants to set up an FHI-aims calculation.
    """
    molecule = MoleculeSpec(
        symbols=symbols,
        coords=coords,
        charge=charge,
        multiplicity=multiplicity,
        lattice_vectors=lattice_vectors,
    )

    xdm_params = None
    if dispersion == "xdm" and xdm_a1 is not None and xdm_a2 is not None:
        xdm_params = XDMParams(a1=xdm_a1, a2=xdm_a2)
    elif dispersion == "xdm":
        xdm_params = XDMParams(keyword=f"bj {basis}")

    params = FHIAimsParams(
        xc=XCFunctional(xc),
        basis=BasisSetTier(basis),
        calculation_type=CalculationType(calculation_type),
        dispersion=DispersionMethod(dispersion) if dispersion != "none" else DispersionMethod.NONE,
        xdm_params=xdm_params,
        k_grid=k_grid,
        compute_forces=compute_forces,
        relativistic=RelativisticTreatment(relativistic),
    )

    gen = FHIAimsGenerator()
    result = gen.generate(molecule, params)

    output_parts = [f"Summary: {result.summary}\n"]
    if result.warnings:
        output_parts.append("Warnings:\n" + "\n".join(f"  - {w}" for w in result.warnings) + "\n")
    for fname, content in result.files.items():
        output_parts.append(f"--- {fname} ---\n{content}")

    return "\n".join(output_parts)
