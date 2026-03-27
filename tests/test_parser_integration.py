"""Integration tests for the output parser using real calculation outputs.

Fixture files come from:
- FHI-aims: GMTKN55 PBE0 benchmarks, H2O relaxation regression test
- Psi4: Pesticides DFT dataset, CCSD(T)/cc-pVTZ benchmarks
- Gaussian: PBE0/aug-cc-pVTZ single-point + force calculations
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chem_copilot.parsers.output_parser import OutputParser

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def parser() -> OutputParser:
    return OutputParser()


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Fixture not found: {path.name}")


# ======================================================================
# FHI-aims
# ======================================================================


class TestFHIAimsHe:
    """He atom PBE0 single-point (GMTKN55/SIE4x4/he)."""

    @pytest.fixture(autouse=True)
    def _check(self) -> None:
        _skip_if_missing(FIXTURES / "fhiaims_he_pbe0.out")

    def test_parse_succeeds(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_he_pbe0.out")
        assert result.program == "fhi-aims"
        assert result.success is True

    def test_total_energy(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_he_pbe0.out")
        assert result.energies.total_energy is not None
        # PBE0 He: ~ -78.78 eV (~ -2.895 Ha)
        assert -80.0 < result.energies.total_energy < -77.0

    def test_hartree_conversion(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_he_pbe0.out")
        ha = result.energies.total_energy_hartree
        assert ha is not None
        assert -3.0 < ha < -2.8

    def test_n_atoms(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_he_pbe0.out")
        assert result.n_atoms == 1

    def test_single_point(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_he_pbe0.out")
        assert result.calc_type == "single_point"


class TestFHIAimsButane:
    """Butane gauche conformer PBE0+XDM (GMTKN55/ACONF/B_G, 14 atoms)."""

    @pytest.fixture(autouse=True)
    def _check(self) -> None:
        _skip_if_missing(FIXTURES / "fhiaims_butane_pbe0.out")

    def test_parse_succeeds(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_butane_pbe0.out")
        assert result.program == "fhi-aims"
        assert result.success is True

    def test_total_energy(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_butane_pbe0.out")
        e = result.energies.total_energy
        assert e is not None
        # PBE0 butane: ~ -4310.95 eV
        assert -4320.0 < e < -4300.0

    def test_n_atoms(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_butane_pbe0.out")
        assert result.n_atoms == 14

    def test_scf_energies(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_butane_pbe0.out")
        assert len(result.energies.scf_energies) >= 1


class TestFHIAimsH2ORelaxation:
    """H2O geometry optimization (PBE, regression test)."""

    @pytest.fixture(autouse=True)
    def _check(self) -> None:
        _skip_if_missing(FIXTURES / "fhiaims_h2o_relaxation.out")

    def test_parse_succeeds(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_h2o_relaxation.out")
        assert result.program == "fhi-aims"
        assert result.success is True

    def test_optimization_detected(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_h2o_relaxation.out")
        assert result.calc_type == "optimization"
        assert result.optimization.n_steps > 1

    def test_optimization_converged(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_h2o_relaxation.out")
        assert result.optimization.converged is True

    def test_final_energy(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_h2o_relaxation.out")
        e = result.energies.total_energy
        assert e is not None
        # PBE H2O: ~ -2078.5 eV
        assert -2080.0 < e < -2077.0

    def test_n_atoms(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_h2o_relaxation.out")
        assert result.n_atoms == 3

    def test_forces_extracted(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "fhiaims_h2o_relaxation.out")
        assert len(result.forces.forces) == 3  # 3 atoms
        assert result.forces.max_force is not None

    def test_energy_decreases(self, parser: OutputParser) -> None:
        """Energy should generally decrease during optimization."""
        result = parser.parse_file(FIXTURES / "fhiaims_h2o_relaxation.out")
        energies = result.optimization.energies
        assert len(energies) > 1
        assert energies[-1] < energies[0]


# ======================================================================
# Psi4
# ======================================================================


class TestPsi4FAtom:
    """F atom wB97M-D3BJ/def2-TZVPPD (pesticides fragment)."""

    @pytest.fixture(autouse=True)
    def _check(self) -> None:
        _skip_if_missing(FIXTURES / "psi4_f_atom_dft.out")

    def test_parse_succeeds(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_f_atom_dft.out", program="psi4")
        assert result.program == "psi4"
        assert result.success is True

    def test_total_energy(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_f_atom_dft.out", program="psi4")
        e = result.energies.total_energy
        assert e is not None
        # wB97M-D3BJ F atom: ~ -2715.3 eV (~ -99.786 Ha)
        assert -2720.0 < e < -2710.0

    def test_n_atoms(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_f_atom_dft.out", program="psi4")
        assert result.n_atoms == 1

    def test_orbitals(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_f_atom_dft.out", program="psi4")
        assert len(result.orbitals.eigenvalues) > 0
        assert result.orbitals.homo_index is not None

    def test_metadata_functional(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_f_atom_dft.out", program="psi4")
        meta = result.raw_metadata.get("metadata", {})
        assert isinstance(meta, dict)
        assert meta.get("functional") == "WB97M-D3BJ"


class TestPsi4HeCCSDT:
    """He CCSD(T)/cc-pVTZ single-point."""

    @pytest.fixture(autouse=True)
    def _check(self) -> None:
        _skip_if_missing(FIXTURES / "psi4_he_ccsdt.out")

    def test_parse_succeeds(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_he_ccsdt.out", program="psi4")
        assert result.program == "psi4"
        assert result.success is True

    def test_ccsd_t_energy(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_he_ccsdt.out", program="psi4")
        # CCSD(T) He: ~ -78.92 eV (~ -2.900 Ha)
        e = result.energies.total_energy
        assert e is not None
        assert -80.0 < e < -78.0

    def test_cc_overrides_scf(self, parser: OutputParser) -> None:
        """CC energy should override SCF as the total energy."""
        result = parser.parse_file(FIXTURES / "psi4_he_ccsdt.out", program="psi4")
        scf = result.energies.scf_energies
        total = result.energies.total_energy
        assert len(scf) > 0
        # CC total should be lower than HF
        assert total is not None
        assert total < scf[-1]

    def test_methods_detected(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_he_ccsdt.out", program="psi4")
        meta = result.raw_metadata.get("metadata", {})
        assert isinstance(meta, dict)
        methods = meta.get("methods", [])
        assert "CCSD(T)" in methods or "CCSD" in methods


class TestPsi4HF2TS:
    """HF dimer TS, CCSD(T)/cc-pVTZ, 4 atoms."""

    @pytest.fixture(autouse=True)
    def _check(self) -> None:
        _skip_if_missing(FIXTURES / "psi4_hf2_ts_ccsdt.out")

    def test_parse_succeeds(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_hf2_ts_ccsdt.out", program="psi4")
        assert result.program == "psi4"
        assert result.success is True

    def test_n_atoms(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_hf2_ts_ccsdt.out", program="psi4")
        assert result.n_atoms == 4

    def test_nbasis(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_hf2_ts_ccsdt.out", program="psi4")
        assert result.n_basis_functions == 88

    def test_orbitals_count(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_hf2_ts_ccsdt.out", program="psi4")
        assert len(result.orbitals.eigenvalues) == 88


class TestPsi4Failed:
    """Psi4 calculation that failed due to SCF convergence."""

    @pytest.fixture(autouse=True)
    def _check(self) -> None:
        _skip_if_missing(FIXTURES / "psi4_scf_failed.out")

    def test_parse_succeeds(self, parser: OutputParser) -> None:
        """Parser should not crash on a failed calculation."""
        result = parser.parse_file(FIXTURES / "psi4_scf_failed.out", program="psi4")
        assert result.program == "psi4"

    def test_no_energy(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_scf_failed.out", program="psi4")
        assert result.energies.total_energy is None

    def test_not_successful(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "psi4_scf_failed.out", program="psi4")
        assert result.success is False


# ======================================================================
# Gaussian
# ======================================================================


class TestGaussianV1:
    """PBE0/aug-cc-pVTZ single-point + force (12 atoms)."""

    @pytest.fixture(autouse=True)
    def _check(self) -> None:
        _skip_if_missing(FIXTURES / "gaussian_v1_pbe0.log")

    def test_auto_detect(self, parser: OutputParser) -> None:
        """Gaussian should be auto-detected without specifying program."""
        result = parser.parse_file(FIXTURES / "gaussian_v1_pbe0.log")
        assert result.program == "gaussian"

    def test_parse_succeeds(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "gaussian_v1_pbe0.log")
        assert result.success is True

    def test_total_energy(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "gaussian_v1_pbe0.log")
        e = result.energies.total_energy
        assert e is not None
        # PBE0/aug-cc-pVTZ, 12 atoms: ~ -6757 eV
        assert -6760.0 < e < -6750.0

    def test_n_atoms(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "gaussian_v1_pbe0.log")
        assert result.n_atoms == 12

    def test_nbasis(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "gaussian_v1_pbe0.log")
        assert result.n_basis_functions == 391

    def test_forces(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "gaussian_v1_pbe0.log")
        assert len(result.forces.forces) == 12
        assert result.forces.max_force is not None
        assert result.forces.max_force > 0

    def test_orbitals(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "gaussian_v1_pbe0.log")
        assert len(result.orbitals.eigenvalues) > 0
        assert result.orbitals.homo_index == 19
        assert result.orbitals.homo_lumo_gap is not None
        assert result.orbitals.homo_lumo_gap > 0

    def test_metadata(self, parser: OutputParser) -> None:
        result = parser.parse_file(FIXTURES / "gaussian_v1_pbe0.log")
        meta = result.raw_metadata.get("metadata", {})
        assert isinstance(meta, dict)
        assert meta.get("basis_set") == "Aug-CC-pVTZ"
        assert meta.get("functional") == "PBE1PBE"


# ======================================================================
# Cross-cutting tests
# ======================================================================


class TestProgramDetection:
    """Verify auto-detection works for each program."""

    def test_fhiaims_detected(self, parser: OutputParser) -> None:
        _skip_if_missing(FIXTURES / "fhiaims_he_pbe0.out")
        result = parser.parse_file(FIXTURES / "fhiaims_he_pbe0.out")
        assert result.program == "fhi-aims"

    def test_gaussian_detected(self, parser: OutputParser) -> None:
        _skip_if_missing(FIXTURES / "gaussian_v1_pbe0.log")
        result = parser.parse_file(FIXTURES / "gaussian_v1_pbe0.log")
        assert result.program == "gaussian"

    def test_psi4_detected(self, parser: OutputParser) -> None:
        _skip_if_missing(FIXTURES / "psi4_he_ccsdt.out")
        result = parser.parse_file(FIXTURES / "psi4_he_ccsdt.out")
        assert result.program == "psi4"
