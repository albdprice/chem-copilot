"""Example: Parse output files from various quantum chemistry codes.

Demonstrates the OutputParser on real calculation outputs.
Run from the project root with test fixtures available.
"""

from pathlib import Path

from chem_copilot.parsers import OutputParser

parser = OutputParser()
fixtures = Path(__file__).parent.parent / "tests" / "fixtures"


def parse_and_show(filepath: Path, program: str | None = None) -> None:
    """Parse a file and print a summary."""
    print(f"File: {filepath.name}")
    r = parser.parse_file(filepath, program=program)
    print(f"  Program: {r.program} | Type: {r.calc_type} | Success: {r.success}")
    if r.n_atoms:
        print(f"  Atoms: {r.n_atoms} | Basis functions: {r.n_basis_functions}")
    e = r.energies
    if e.total_energy is not None:
        print(f"  Total energy: {e.total_energy:.6f} eV ({e.total_energy_hartree:.8f} Ha)")
    if e.scf_energies:
        print(f"  SCF iterations: {len(e.scf_energies)}")
    if r.orbitals.homo_lumo_gap is not None:
        print(f"  HOMO-LUMO gap: {r.orbitals.homo_lumo_gap:.4f} eV")
    if r.forces.max_force is not None:
        print(f"  Max force: {r.forces.max_force:.6f} eV/Ang")
    if r.optimization.n_steps > 1:
        print(f"  Optimization: {r.optimization.n_steps} steps, converged={r.optimization.converged}")
    print()


# --- FHI-aims ---
print("=" * 60)
print("FHI-aims outputs")
print("=" * 60)

for name in ["fhiaims_he_pbe0.out", "fhiaims_butane_pbe0.out", "fhiaims_h2o_relaxation.out"]:
    f = fixtures / name
    if f.exists():
        parse_and_show(f)

# --- Psi4 ---
print("=" * 60)
print("Psi4 outputs")
print("=" * 60)

for name, prog in [
    ("psi4_f_atom_dft.out", "psi4"),
    ("psi4_he_ccsdt.out", "psi4"),
    ("psi4_hf2_ts_ccsdt.out", "psi4"),
    ("psi4_scf_failed.out", "psi4"),
]:
    f = fixtures / name
    if f.exists():
        parse_and_show(f, program=prog)

# --- Gaussian ---
print("=" * 60)
print("Gaussian outputs")
print("=" * 60)

f = fixtures / "gaussian_v1_pbe0.log"
if f.exists():
    parse_and_show(f)
