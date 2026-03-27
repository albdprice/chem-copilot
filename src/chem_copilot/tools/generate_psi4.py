"""LangGraph tool: Psi4 input file generation."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from chem_copilot.generators.psi4 import Psi4Generator
from chem_copilot.models import (
    CalculationType,
    DispersionMethod,
    MoleculeSpec,
    Psi4Method,
    Psi4Params,
)


class Psi4Input(BaseModel):
    """Input schema for the Psi4 generator tool."""

    symbols: list[str] = Field(description="Element symbols, e.g. ['O', 'H', 'H']")
    coords: list[list[float]] = Field(
        description="Cartesian coordinates in Angstrom, shape (N, 3)"
    )
    charge: int = Field(default=0, description="Net charge")
    multiplicity: int = Field(default=1, description="Spin multiplicity (2S+1)")
    method: str = Field(default="hf", description="Method: hf, dft, mp2, ccsd, ccsd(t)")
    functional: Optional[str] = Field(
        default=None, description="DFT functional (required when method=dft), e.g. b3lyp, pbe"
    )
    basis: str = Field(default="cc-pVDZ", description="Basis set, e.g. cc-pVTZ, aug-cc-pVTZ, def2-TZVPPD")
    calculation_type: str = Field(
        default="single_point",
        description="Calculation type: single_point, optimization, frequency",
    )
    dispersion: str = Field(default="none", description="Dispersion: none, d3, d3bj, d4")
    frozen_core: bool = Field(default=True, description="Freeze core orbitals in correlated methods")
    memory: str = Field(default="4 GB", description="Memory allocation")


@tool("generate_psi4_input", args_schema=Psi4Input)
def generate_psi4_input(
    symbols: list[str],
    coords: list[list[float]],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "hf",
    functional: Optional[str] = None,
    basis: str = "cc-pVDZ",
    calculation_type: str = "single_point",
    dispersion: str = "none",
    frozen_core: bool = True,
    memory: str = "4 GB",
) -> str:
    """Generate a Psi4 Python input script.

    Given a molecular system and calculation parameters, produces a
    ready-to-run Psi4 input script. Returns the script content and
    a summary.

    Use this tool when the user wants to set up a Psi4 calculation.
    """
    molecule = MoleculeSpec(
        symbols=symbols, coords=coords,
        charge=charge, multiplicity=multiplicity,
    )

    params = Psi4Params(
        method=Psi4Method(method),
        functional=functional,
        basis=basis,
        calculation_type=CalculationType(calculation_type),
        dispersion=DispersionMethod(dispersion) if dispersion != "none" else DispersionMethod.NONE,
        frozen_core=frozen_core,
        memory=memory,
    )

    gen = Psi4Generator()
    result = gen.generate(molecule, params)

    output_parts = [f"Summary: {result.summary}\n"]
    if result.warnings:
        output_parts.append("Warnings:\n" + "\n".join(f"  - {w}" for w in result.warnings) + "\n")
    for fname, content in result.files.items():
        output_parts.append(f"--- {fname} ---\n{content}")

    return "\n".join(output_parts)
