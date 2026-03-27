"""LangGraph tool: Parse computational chemistry output files."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from chem_copilot.parsers.output_parser import OutputParser


class ParseOutputInput(BaseModel):
    """Input schema for the output parser tool."""

    filepath: str = Field(description="Path to the output file to parse")
    program: Optional[str] = Field(
        default=None,
        description="Program hint: 'fhi-aims', 'psi4', or 'gaussian'. Auto-detected if not given.",
    )


@tool("parse_calculation_output", args_schema=ParseOutputInput)
def parse_calculation_output(
    filepath: str,
    program: Optional[str] = None,
) -> str:
    """Parse a computational chemistry output file and extract key results.

    Extracts energies, forces, orbital energies, frequencies, and
    optimization trajectories from FHI-aims, Psi4, or Gaussian output files.

    Use this tool when the user wants to analyze results from a completed calculation.
    """
    parser = OutputParser()
    result = parser.parse_file(filepath, program=program)

    lines = [
        f"Program: {result.program}",
        f"Calculation type: {result.calc_type}",
        f"Success: {result.success}",
    ]

    if result.n_atoms is not None:
        lines.append(f"Atoms: {result.n_atoms}")
    if result.n_basis_functions is not None:
        lines.append(f"Basis functions: {result.n_basis_functions}")

    # Energies
    e = result.energies
    if e.total_energy is not None:
        lines.append(f"\nTotal energy: {e.total_energy:.6f} eV ({e.total_energy_hartree:.8f} Ha)")
    if e.scf_energies:
        lines.append(f"SCF iterations: {len(e.scf_energies)}")
    if e.ccsd_t_energy is not None:
        lines.append(f"CCSD(T) energy: {e.ccsd_t_energy:.8f} Ha")
    if e.mp2_energy is not None:
        lines.append(f"MP2 energy: {e.mp2_energy:.8f} Ha")

    # Orbitals
    orb = result.orbitals
    if orb.homo_energy is not None:
        lines.append(f"\nHOMO energy: {orb.homo_energy:.4f} eV")
    if orb.lumo_energy is not None:
        lines.append(f"LUMO energy: {orb.lumo_energy:.4f} eV")
    if orb.homo_lumo_gap is not None:
        lines.append(f"HOMO-LUMO gap: {orb.homo_lumo_gap:.4f} eV")

    # Forces
    if result.forces.max_force is not None:
        lines.append(f"\nMax force: {result.forces.max_force:.6f} eV/Ang")
        lines.append(f"RMS force: {result.forces.rms_force:.6f} eV/Ang")

    # Optimization
    opt = result.optimization
    if opt.n_steps > 1:
        lines.append(f"\nOptimization steps: {opt.n_steps}")
        lines.append(f"Converged: {opt.converged}")
        if opt.energies:
            lines.append(f"Energy change: {opt.energies[-1] - opt.energies[0]:.6f} eV")

    # Frequencies
    if result.frequencies.frequencies:
        lines.append(f"\nVibrational modes: {len(result.frequencies.frequencies)}")
        lines.append(f"Imaginary modes: {result.frequencies.n_imaginary}")

    if result.warnings:
        lines.append("\nWarnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)
