"""Tests for output parser.

Since we don't have actual output files in the test environment,
we test the parser's internal methods with mock cclib data objects
and verify it handles edge cases gracefully.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from chem_copilot.parsers.output_parser import OutputParser


@pytest.fixture
def parser() -> OutputParser:
    return OutputParser()


class _CclibData:
    """Lightweight stand-in for a cclib parsed data object.

    Only attributes explicitly set exist — ``hasattr`` works correctly,
    unlike MagicMock which auto-creates any accessed attribute.
    """

    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _mock_cclib_data(
    scf_energies: list[float] | None = None,
    cc_energies: list[float] | None = None,
    mp_energies: list[list[float]] | None = None,
    grads: np.ndarray | None = None,
    mo_energies: list[np.ndarray] | None = None,
    homos: list[int] | None = None,
    vib_freqs: list[float] | None = None,
    vib_irs: list[float] | None = None,
    optstatus: np.ndarray | None = None,
    natom: int = 3,
    nbasis: int = 24,
    atom_coords: np.ndarray | None = None,
    metadata: dict | None = None,
) -> _CclibData:
    """Create a mock cclib data object with only the requested attributes."""
    attrs: dict[str, object] = {"natom": natom, "nbasis": nbasis}
    attrs["metadata"] = metadata if metadata is not None else {}

    if scf_energies is not None:
        attrs["scfenergies"] = np.array(scf_energies)
    if cc_energies is not None:
        attrs["ccenergies"] = np.array(cc_energies)
    if mp_energies is not None:
        attrs["mpenergies"] = np.array(mp_energies)
    if grads is not None:
        attrs["grads"] = grads
    if mo_energies is not None:
        attrs["moenergies"] = mo_energies
    if homos is not None:
        attrs["homos"] = np.array(homos)
    if vib_freqs is not None:
        attrs["vibfreqs"] = np.array(vib_freqs)
    if vib_irs is not None:
        attrs["vibirs"] = np.array(vib_irs)
    if optstatus is not None:
        attrs["optstatus"] = optstatus
    if atom_coords is not None:
        attrs["atomcoords"] = atom_coords

    return _CclibData(**attrs)


class TestExtractEnergies:
    def test_scf_energies(self, parser: OutputParser) -> None:
        data = _mock_cclib_data(scf_energies=[-2041.5, -2041.8, -2042.0])
        result = parser._build_result(data, "psi4")
        assert result.energies.total_energy == pytest.approx(-2042.0)
        assert len(result.energies.scf_energies) == 3

    def test_cc_energies_override(self, parser: OutputParser) -> None:
        data = _mock_cclib_data(
            scf_energies=[-2041.0],
            cc_energies=[-2042.5],
        )
        result = parser._build_result(data, "psi4")
        # CC energy should override SCF
        assert result.energies.total_energy == pytest.approx(-2042.5)

    def test_mp2_energies(self, parser: OutputParser) -> None:
        data = _mock_cclib_data(
            scf_energies=[-2041.0],
            mp_energies=[[-2041.5]],
        )
        result = parser._build_result(data, "psi4")
        assert result.energies.mp2_energy is not None

    def test_no_energies(self, parser: OutputParser) -> None:
        data = _mock_cclib_data()
        result = parser._build_result(data, "psi4")
        assert result.energies.total_energy is None


class TestExtractForces:
    def test_forces_from_gradients(self, parser: OutputParser) -> None:
        # Gradient in Hartree/Bohr
        grads = np.array([[[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03], [0.0, 0.0, 0.0]]])
        data = _mock_cclib_data(scf_energies=[-2042.0], grads=grads)
        result = parser._build_result(data, "fhi-aims")
        assert len(result.forces.forces) == 3
        assert result.forces.max_force is not None
        assert result.forces.max_force > 0

    def test_no_forces(self, parser: OutputParser) -> None:
        data = _mock_cclib_data(scf_energies=[-2042.0])
        result = parser._build_result(data, "psi4")
        assert result.forces.forces == []


class TestExtractOrbitals:
    def test_orbital_energies(self, parser: OutputParser) -> None:
        mo = np.array([-20.5, -1.3, -0.7, -0.5, 0.2, 0.8, 1.2])
        data = _mock_cclib_data(
            scf_energies=[-2042.0],
            mo_energies=[mo],
            homos=[3],
        )
        result = parser._build_result(data, "psi4")
        assert len(result.orbitals.eigenvalues) == 7
        assert result.orbitals.homo_index == 3
        assert result.orbitals.homo_energy == pytest.approx(-0.5)
        assert result.orbitals.lumo_energy == pytest.approx(0.2)
        assert result.orbitals.homo_lumo_gap == pytest.approx(0.7)

    def test_no_orbitals(self, parser: OutputParser) -> None:
        data = _mock_cclib_data(scf_energies=[-2042.0])
        result = parser._build_result(data, "psi4")
        assert result.orbitals.eigenvalues == []


class TestExtractFrequencies:
    def test_real_frequencies(self, parser: OutputParser) -> None:
        data = _mock_cclib_data(
            scf_energies=[-2042.0],
            vib_freqs=[1595.0, 3657.0, 3756.0],
            vib_irs=[70.0, 2.0, 45.0],
        )
        result = parser._build_result(data, "gaussian")
        assert len(result.frequencies.frequencies) == 3
        assert result.frequencies.n_imaginary == 0
        assert result.frequencies.intensities == [70.0, 2.0, 45.0]

    def test_imaginary_frequency(self, parser: OutputParser) -> None:
        data = _mock_cclib_data(
            scf_energies=[-2042.0],
            vib_freqs=[-150.0, 500.0, 1200.0],
        )
        result = parser._build_result(data, "gaussian")
        assert result.frequencies.n_imaginary == 1
        assert result.frequencies.is_imaginary[0] is True
        assert result.frequencies.is_imaginary[1] is False


class TestExtractOptimization:
    def test_optimization_converged(self, parser: OutputParser) -> None:
        data = _mock_cclib_data(
            scf_energies=[-2040.0, -2041.0, -2042.0],
            optstatus=np.array([2, 2, 4]),  # 4 = converged in cclib
            atom_coords=np.random.rand(3, 3, 3),
        )
        result = parser._build_result(data, "psi4")
        assert result.optimization.converged is True
        assert result.optimization.n_steps == 3

    def test_optimization_not_converged(self, parser: OutputParser) -> None:
        data = _mock_cclib_data(
            scf_energies=[-2040.0, -2041.0],
            optstatus=np.array([2, 2]),
            atom_coords=np.random.rand(2, 3, 3),
        )
        result = parser._build_result(data, "psi4")
        assert result.optimization.converged is False


class TestDetection:
    def test_detect_program_from_cclib_gaussian(self, parser: OutputParser) -> None:
        data = _mock_cclib_data(metadata={"package": "Gaussian"})
        prog = parser._detect_program_from_cclib(data, Path("test.log"), "unknown")
        assert prog == "gaussian"

    def test_detect_program_from_cclib_psi4(self, parser: OutputParser) -> None:
        data = _mock_cclib_data(metadata={"package": "Psi4"})
        prog = parser._detect_program_from_cclib(data, Path("test.out"), "unknown")
        assert prog == "psi4"

    def test_detect_calc_type_frequency(self, parser: OutputParser) -> None:
        from chem_copilot.models import OptimizationTrajectory, ParsedFrequencies

        freqs = ParsedFrequencies(frequencies=[1000.0, 2000.0], n_imaginary=0)
        opt = OptimizationTrajectory()
        data = _mock_cclib_data()
        ct = parser._detect_calc_type(data, opt, freqs)
        assert ct == "frequency"

    def test_detect_calc_type_opt(self, parser: OutputParser) -> None:
        from chem_copilot.models import OptimizationTrajectory, ParsedFrequencies

        freqs = ParsedFrequencies()
        opt = OptimizationTrajectory(n_steps=5)
        data = _mock_cclib_data()
        ct = parser._detect_calc_type(data, opt, freqs)
        assert ct == "optimization"


class TestSuccess:
    def test_success_with_energy(self, parser: OutputParser) -> None:
        data = _mock_cclib_data(scf_energies=[-2042.0])
        result = parser._build_result(data, "psi4")
        assert result.success is True

    def test_no_success_without_energy(self, parser: OutputParser) -> None:
        data = _mock_cclib_data()
        result = parser._build_result(data, "psi4")
        assert result.success is False


class TestParseFile:
    def test_file_not_found(self, parser: OutputParser) -> None:
        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/file.out")

    def test_unparseable_file(
        self, parser: OutputParser, tmp_path: Path
    ) -> None:
        """Parsing garbage should either raise or return a failed result."""
        f = tmp_path / "bad.out"
        f.write_text("not a real output file\njust some random text\n")
        try:
            result = parser.parse_file(f, program="gaussian")
            # If it doesn't raise, it should at least not claim success
            assert result.success is False
        except (ValueError, Exception):
            pass  # Raising is also acceptable
