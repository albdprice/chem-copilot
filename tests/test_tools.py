"""Tests for LangGraph tool wrappers."""

from __future__ import annotations

import pytest

from chem_copilot.tools import (
    ALL_TOOLS,
    generate_fhiaims_input,
    generate_psi4_input,
    parse_calculation_output,
)


class TestFHIAimsTool:
    def test_basic_invocation(self) -> None:
        result = generate_fhiaims_input.invoke({
            "symbols": ["O", "H", "H"],
            "coords": [
                [0.0, 0.0, 0.1173],
                [0.0, 0.7572, -0.4692],
                [0.0, -0.7572, -0.4692],
            ],
        })
        assert "control.in" in result
        assert "geometry.in" in result
        assert "xc" in result
        assert "H2O" in result

    def test_pbe0_xdm_tight(self) -> None:
        result = generate_fhiaims_input.invoke({
            "symbols": ["O", "H", "H"],
            "coords": [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
            "xc": "pbe0",
            "basis": "tight",
            "dispersion": "xdm",
        })
        assert "pbe0" in result
        assert "xdm bj tight" in result

    def test_periodic_system(self) -> None:
        result = generate_fhiaims_input.invoke({
            "symbols": ["Si", "Si"],
            "coords": [[0.0, 0.0, 0.0], [1.354, 1.354, 1.354]],
            "lattice_vectors": [[0.0, 2.708, 2.708], [2.708, 0.0, 2.708], [2.708, 2.708, 0.0]],
            "k_grid": [4, 4, 4],
        })
        assert "k_grid" in result
        assert "lattice_vector" in result

    def test_spin_polarized(self) -> None:
        result = generate_fhiaims_input.invoke({
            "symbols": ["O", "O"],
            "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2075]],
            "multiplicity": 3,
        })
        assert "collinear" in result


class TestPsi4Tool:
    def test_basic_invocation(self) -> None:
        result = generate_psi4_input.invoke({
            "symbols": ["O", "H", "H"],
            "coords": [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
        })
        assert "input.py" in result
        assert "import psi4" in result
        assert "hf" in result

    def test_ccsd_t(self) -> None:
        result = generate_psi4_input.invoke({
            "symbols": ["O", "H", "H"],
            "coords": [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
            "method": "ccsd(t)",
            "basis": "cc-pVTZ",
        })
        assert "ccsd(t)" in result
        assert "cc-pVTZ" in result

    def test_dft_b3lyp_d3bj(self) -> None:
        result = generate_psi4_input.invoke({
            "symbols": ["C", "H", "H", "H", "H"],
            "coords": [
                [0.0, 0.0, 0.0], [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63],
                [0.63, -0.63, -0.63],
            ],
            "method": "dft",
            "functional": "b3lyp",
            "dispersion": "d3bj",
        })
        assert "b3lyp-d3bj" in result


class TestParseOutputTool:
    def test_fhiaims_output(self) -> None:
        from pathlib import Path
        fixture = Path(__file__).parent / "fixtures" / "fhiaims_he_pbe0.out"
        if not fixture.exists():
            pytest.skip("Fixture not available")
        result = parse_calculation_output.invoke({
            "filepath": str(fixture),
            "program": "fhi-aims",
        })
        assert "Total energy" in result
        assert "fhi-aims" in result.lower()

    def test_gaussian_output(self) -> None:
        from pathlib import Path
        fixture = Path(__file__).parent / "fixtures" / "gaussian_v1_pbe0.log"
        if not fixture.exists():
            pytest.skip("Fixture not available")
        result = parse_calculation_output.invoke({
            "filepath": str(fixture),
        })
        assert "gaussian" in result.lower()
        assert "Total energy" in result


class TestToolList:
    def test_all_tools_present(self) -> None:
        assert len(ALL_TOOLS) == 9
        names = {t.name for t in ALL_TOOLS}
        assert "generate_fhiaims_input" in names
        assert "generate_psi4_input" in names
        assert "parse_calculation_output" in names
        assert "submit_fhiaims_job" in names
        assert "submit_psi4_job" in names
        assert "check_job_status" in names
        assert "get_job_result" in names

    def test_tools_have_descriptions(self) -> None:
        for t in ALL_TOOLS:
            assert t.description
            assert len(t.description) > 20
