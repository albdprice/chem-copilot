"""Tests for Psi4 input generator."""

from __future__ import annotations

import pytest

from chem_copilot.generators.psi4 import Psi4Generator
from chem_copilot.models import (
    CalculationType,
    DispersionMethod,
    MoleculeSpec,
    Psi4Method,
    Psi4Params,
)


@pytest.fixture
def gen() -> Psi4Generator:
    return Psi4Generator()


class TestPsi4Script:
    """Tests for Psi4 input script generation."""

    def test_default_hf(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        result = gen.generate(water)
        script = result.files["input.py"]
        assert "import psi4" in script
        assert "'basis': 'cc-pVDZ'" in script
        assert "'reference': 'rhf'" in script
        assert "psi4.energy('hf')" in script

    def test_dft_b3lyp(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        params = Psi4Params(method=Psi4Method.DFT, functional="b3lyp")
        script = gen.generate(water, params).files["input.py"]
        assert "psi4.energy('b3lyp')" in script

    def test_dft_with_d3bj(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        params = Psi4Params(
            method=Psi4Method.DFT,
            functional="b3lyp",
            dispersion=DispersionMethod.D3BJ,
        )
        script = gen.generate(water, params).files["input.py"]
        assert "b3lyp-d3bj" in script

    def test_mp2(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        params = Psi4Params(
            method=Psi4Method.MP2, basis="aug-cc-pVTZ"
        )
        script = gen.generate(water, params).files["input.py"]
        assert "psi4.energy('mp2')" in script
        assert "'basis': 'aug-cc-pVTZ'" in script
        assert "'freeze_core': 'true'" in script

    def test_ccsd_t(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        params = Psi4Params(method=Psi4Method.CCSD_T, basis="cc-pVTZ")
        script = gen.generate(water, params).files["input.py"]
        assert "psi4.energy('ccsd(t)')" in script
        assert "'freeze_core': 'true'" in script

    def test_no_frozen_core(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        params = Psi4Params(method=Psi4Method.MP2, frozen_core=False)
        script = gen.generate(water, params).files["input.py"]
        assert "'freeze_core': 'false'" in script

    def test_optimization(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        params = Psi4Params(calculation_type=CalculationType.OPTIMIZATION)
        script = gen.generate(water, params).files["input.py"]
        assert "psi4.optimize('hf'" in script
        assert "return_wfn=True" in script

    def test_frequency(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        params = Psi4Params(calculation_type=CalculationType.FREQUENCY)
        script = gen.generate(water, params).files["input.py"]
        assert "psi4.frequency('hf'" in script

    def test_open_shell_uhf(
        self, gen: Psi4Generator, oxygen_triplet: MoleculeSpec
    ) -> None:
        result = gen.generate(oxygen_triplet)
        script = result.files["input.py"]
        assert "'reference': 'uhf'" in script

    def test_open_shell_cc_rohf(
        self, gen: Psi4Generator, oxygen_triplet: MoleculeSpec
    ) -> None:
        params = Psi4Params(method=Psi4Method.CCSD_T)
        result = gen.generate(oxygen_triplet, params)
        script = result.files["input.py"]
        assert "'reference': 'rohf'" in script
        assert any("rohf" in w.lower() for w in result.warnings)

    def test_charge_multiplicity_line(
        self, gen: Psi4Generator, water_charged: MoleculeSpec
    ) -> None:
        script = gen.generate(water_charged).files["input.py"]
        assert "-1 1" in script

    def test_periodic_warning(
        self, gen: Psi4Generator, nacl_crystal: MoleculeSpec
    ) -> None:
        result = gen.generate(nacl_crystal)
        assert any("periodic" in w.lower() for w in result.warnings)

    def test_symmetry(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        params = Psi4Params(symmetry="c2v")
        script = gen.generate(water, params).files["input.py"]
        assert "symmetry c2v" in script

    def test_memory_and_threads(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        params = Psi4Params(memory="8 GB", num_threads=16)
        script = gen.generate(water, params).files["input.py"]
        assert "psi4.set_memory('8 GB')" in script
        assert "psi4.set_num_threads(16)" in script

    def test_no_reorient_no_com(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        script = gen.generate(water).files["input.py"]
        assert "no_reorient" in script
        assert "no_com" in script

    def test_extra_options(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        params = Psi4Params(extra_options={"print": "2"})
        script = gen.generate(water, params).files["input.py"]
        assert "'print': '2'" in script

    def test_xyz_input(
        self, gen: Psi4Generator, water_xyz: MoleculeSpec
    ) -> None:
        result = gen.generate(water_xyz)
        script = result.files["input.py"]
        assert "O" in script
        assert "H" in script


class TestSummary:
    def test_summary_basic(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        result = gen.generate(water)
        assert "H2O" in result.summary
        assert "hf" in result.summary
        assert "cc-pVDZ" in result.summary

    def test_summary_charged(
        self, gen: Psi4Generator, water_charged: MoleculeSpec
    ) -> None:
        result = gen.generate(water_charged)
        assert "charge" in result.summary

    def test_summary_spin(
        self, gen: Psi4Generator, oxygen_triplet: MoleculeSpec
    ) -> None:
        result = gen.generate(oxygen_triplet)
        assert "mult" in result.summary


class TestGeneratedFiles:
    def test_produces_one_file(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        result = gen.generate(water)
        assert set(result.files.keys()) == {"input.py"}

    def test_file_ends_with_newline(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        result = gen.generate(water)
        assert result.files["input.py"].endswith("\n")

    def test_valid_python_syntax(
        self, gen: Psi4Generator, water: MoleculeSpec
    ) -> None:
        script = gen.generate(water).files["input.py"]
        compile(script, "<test>", "exec")  # Raises SyntaxError if invalid
