"""Tests for FHI-aims input generator.

Many of the expected strings are validated against the FHI-aims regression
test suite (regression_tests/testcases/) to ensure our output matches
real FHI-aims conventions.
"""

from __future__ import annotations

import pytest

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


@pytest.fixture
def gen() -> FHIAimsGenerator:
    return FHIAimsGenerator()


@pytest.fixture
def diamond() -> MoleculeSpec:
    """Diamond unit cell (matches xdm-diamond regression test)."""
    return MoleculeSpec(
        symbols=["C", "C"],
        coords=[
            [0.4458375, 0.4458375, 0.4458375],
            [3.1208625, 3.1208625, 3.1208625],
        ],
        lattice_vectors=[
            [0.0, 1.78335, 1.78335],
            [1.78335, 0.0, 1.78335],
            [1.78335, 1.78335, 0.0],
        ],
    )


@pytest.fixture
def water_dimer() -> MoleculeSpec:
    """Water dimer (matches xdm-h2o-dimer regression test)."""
    return MoleculeSpec(
        symbols=["O", "H", "H", "O", "H", "H"],
        coords=[
            [10.3832520, 10.2592210, 10.7585610],
            [10.0000000, 11.1362440, 10.7585610],
            [11.3345820, 10.4144530, 10.7585610],
            [13.2848840, 10.4852100, 10.7585610],
            [13.6146570, 10.0000000, 10.0000000],
            [13.6146570, 10.0000000, 11.5171220],
        ],
    )


class TestControlIn:
    """Tests for control.in generation."""

    def test_default_pbe_light(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        result = gen.generate(water)
        ctrl = result.files["control.in"]
        assert "xc                 pbe" in ctrl
        assert "relativistic       atomic_zora scalar" in ctrl
        assert "spin               none" in ctrl
        assert "sc_accuracy_rho" in ctrl
        assert "Basis set: light" in ctrl

    def test_hybrid_functional(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(xc=XCFunctional.PBE0, basis=BasisSetTier.TIGHT)
        result = gen.generate(water, params)
        ctrl = result.files["control.in"]
        assert "xc                 pbe0" in ctrl
        assert "Basis set: tight" in ctrl

    def test_hse06_with_unit(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        """HSE06 needs omega=0.11 and hse_unit bohr-1 (cf. Si.hse06 regression test)."""
        params = FHIAimsParams(xc=XCFunctional.HSE06)
        ctrl = gen.generate(water, params).files["control.in"]
        assert "hse06  0.11" in ctrl
        assert "hse_unit bohr-1" in ctrl

    def test_xc_pre(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        """xc_pre preconverges with cheaper functional (cf. Si.hse06.xc_pre)."""
        params = FHIAimsParams(
            xc=XCFunctional.HSE06,
            xc_pre=("pbe", 10),
        )
        ctrl = gen.generate(water, params).files["control.in"]
        assert "xc_pre             pbe 10" in ctrl

    def test_b86bpbe(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        """B86bPBE is the standard XDM functional."""
        params = FHIAimsParams(xc=XCFunctional.B86BPBE)
        ctrl = gen.generate(water, params).files["control.in"]
        assert "xc                 b86bpbe" in ctrl

    # --- Dispersion corrections ---

    def test_dispersion_ts(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(dispersion=DispersionMethod.TS)
        ctrl = gen.generate(water, params).files["control.in"]
        assert "vdw_correction_hirshfeld" in ctrl

    def test_dispersion_mbd(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        """MBD keyword (cf. mbd-h2-dimer: many_body_dispersion)."""
        params = FHIAimsParams(dispersion=DispersionMethod.MBD)
        ctrl = gen.generate(water, params).files["control.in"]
        assert "many_body_dispersion" in ctrl
        # Should NOT contain "many_body_dispersion_nl"
        assert "many_body_dispersion_nl" not in ctrl

    def test_dispersion_mbd_nl(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(dispersion=DispersionMethod.MBD_NL)
        ctrl = gen.generate(water, params).files["control.in"]
        assert "many_body_dispersion_nl" in ctrl

    def test_xdm_explicit_damping(
        self, gen: FHIAimsGenerator, water_dimer: MoleculeSpec
    ) -> None:
        """XDM with explicit a1/a2 (cf. xdm-diamond: xdm 0.69125110 1.57470830)."""
        params = FHIAimsParams(
            xc=XCFunctional.B86BPBE,
            dispersion=DispersionMethod.XDM,
            xdm_params=XDMParams(a1=0.69125110, a2=1.57470830),
        )
        ctrl = gen.generate(water_dimer, params).files["control.in"]
        assert "xdm 0.69125110 1.57470830" in ctrl

    def test_xdm_bj_keyword(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        """XDM with bj keyword (cf. GMTKN55: xdm bj tight)."""
        params = FHIAimsParams(
            xc=XCFunctional.PBE0,
            basis=BasisSetTier.TIGHT,
            dispersion=DispersionMethod.XDM,
            xdm_params=XDMParams(keyword="bj tight"),
        )
        ctrl = gen.generate(water, params).files["control.in"]
        assert "xdm bj tight" in ctrl

    def test_xdm_lightdense(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        """XDM with tier keyword (cf. xdm-graphite: xdm lightdense)."""
        params = FHIAimsParams(
            dispersion=DispersionMethod.XDM,
            xdm_params=XDMParams(keyword="lightdense"),
        )
        ctrl = gen.generate(water, params).files["control.in"]
        assert "xdm lightdense" in ctrl

    def test_xdm_default_uses_basis_tier(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        """XDM with no explicit params defaults to 'xdm bj <tier>'."""
        params = FHIAimsParams(
            basis=BasisSetTier.TIGHT,
            dispersion=DispersionMethod.XDM,
        )
        ctrl = gen.generate(water, params).files["control.in"]
        assert "xdm bj tight" in ctrl

    def test_d3_unsupported_warning(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(dispersion=DispersionMethod.D3)
        result = gen.generate(water, params)
        assert any("not natively" in w for w in result.warnings)

    # --- Spin ---

    def test_spin_collinear_auto(
        self, gen: FHIAimsGenerator, oxygen_triplet: MoleculeSpec
    ) -> None:
        result = gen.generate(oxygen_triplet)
        ctrl = result.files["control.in"]
        assert "spin               collinear" in ctrl
        assert "default_initial_moment 2" in ctrl

    def test_spin_none_for_singlet(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        ctrl = gen.generate(water).files["control.in"]
        assert "spin               none" in ctrl

    # --- Charge ---

    def test_charge(
        self, gen: FHIAimsGenerator, water_charged: MoleculeSpec
    ) -> None:
        ctrl = gen.generate(water_charged).files["control.in"]
        assert "charge             -1" in ctrl

    # --- Relaxation / forces ---

    def test_optimization(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(calculation_type=CalculationType.OPTIMIZATION)
        ctrl = gen.generate(water, params).files["control.in"]
        assert "relax_geometry" in ctrl
        assert "sc_accuracy_forces" in ctrl

    def test_compute_forces(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        """compute_forces .true. (cf. xdm-diamond regression test)."""
        params = FHIAimsParams(compute_forces=True)
        ctrl = gen.generate(water, params).files["control.in"]
        assert "compute_forces .true." in ctrl
        assert "sc_accuracy_forces" in ctrl

    def test_compute_analytical_stress(
        self, gen: FHIAimsGenerator, diamond: MoleculeSpec
    ) -> None:
        """compute_analytical_stress for periodic (cf. xdm-diamond)."""
        params = FHIAimsParams(
            k_grid=[2, 2, 2],
            compute_analytical_stress=True,
        )
        ctrl = gen.generate(diamond, params).files["control.in"]
        assert "compute_analytical_stress .true." in ctrl

    # --- Periodic ---

    def test_periodic_k_grid(
        self, gen: FHIAimsGenerator, nacl_crystal: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(k_grid=[4, 4, 4])
        ctrl = gen.generate(nacl_crystal, params).files["control.in"]
        assert "k_grid             4 4 4" in ctrl

    def test_periodic_default_k_grid_warning(
        self, gen: FHIAimsGenerator, nacl_crystal: MoleculeSpec
    ) -> None:
        result = gen.generate(nacl_crystal)
        assert any("k_grid" in w for w in result.warnings)
        assert "k_grid             1 1 1" in result.files["control.in"]

    def test_k_grid_non_periodic_warning(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(k_grid=[4, 4, 4])
        result = gen.generate(water, params)
        assert any("non-periodic" in w for w in result.warnings)

    # --- MD ---

    def test_md_parameters(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(
            calculation_type=CalculationType.MOLECULAR_DYNAMICS,
            md_ensemble="NVT_berendsen",
            md_temperature=300.0,
            md_time_step=0.5,
            md_n_steps=1000,
        )
        ctrl = gen.generate(water, params).files["control.in"]
        assert "MD_run" in ctrl
        assert "NVT_berendsen" in ctrl
        assert "300.0" in ctrl
        assert "0.5" in ctrl

    # --- Output / extras ---

    def test_output_requests(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(output_requests=["mulliken", "hirshfeld"])
        ctrl = gen.generate(water, params).files["control.in"]
        assert "output             mulliken" in ctrl
        assert "output             hirshfeld" in ctrl

    def test_extra_keywords(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(extra_keywords={"mixer": "pulay"})
        ctrl = gen.generate(water, params).files["control.in"]
        assert "mixer" in ctrl
        assert "pulay" in ctrl

    def test_species_defaults_paths(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        ctrl = gen.generate(water).files["control.in"]
        assert "08_O_default" in ctrl
        assert "01_H_default" in ctrl

    def test_scan_functional(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(xc=XCFunctional.SCAN)
        ctrl = gen.generate(water, params).files["control.in"]
        assert "dfauto scan" in ctrl


class TestRegressionTestComparison:
    """Compare generated output against known FHI-aims regression test conventions."""

    def test_xdm_diamond_control_keywords(
        self, gen: FHIAimsGenerator, diamond: MoleculeSpec
    ) -> None:
        """Validate against xdm-diamond regression test."""
        params = FHIAimsParams(
            xc=XCFunctional.B86BPBE,
            dispersion=DispersionMethod.XDM,
            xdm_params=XDMParams(a1=0.69125110, a2=1.57470830),
            compute_forces=True,
            compute_analytical_stress=True,
            k_grid=[2, 2, 2],
        )
        result = gen.generate(diamond, params)
        ctrl = result.files["control.in"]

        # Match key lines from the regression test control.in
        assert "xc                 b86bpbe" in ctrl
        assert "xdm 0.69125110 1.57470830" in ctrl
        assert "compute_forces .true." in ctrl
        assert "compute_analytical_stress .true." in ctrl
        assert "k_grid             2 2 2" in ctrl

    def test_xdm_h2o_dimer_control(
        self, gen: FHIAimsGenerator, water_dimer: MoleculeSpec
    ) -> None:
        """Validate against xdm-h2o-dimer regression test."""
        params = FHIAimsParams(
            xc=XCFunctional.B86BPBE,
            dispersion=DispersionMethod.XDM,
            xdm_params=XDMParams(a1=0.69125110, a2=1.57470830),
            compute_forces=True,
        )
        result = gen.generate(water_dimer, params)
        ctrl = result.files["control.in"]
        assert "xdm 0.69125110 1.57470830" in ctrl
        assert "compute_forces .true." in ctrl
        assert "spin               none" in ctrl

    def test_h2o_relaxation_style(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        """Matches H2O.pbe.relaxation regression test conventions."""
        params = FHIAimsParams(
            xc=XCFunctional.PBE,
            relativistic=RelativisticTreatment.NONE,
            calculation_type=CalculationType.OPTIMIZATION,
            relax_geometry="bfgs 1e-2",
            occupation_type="gaussian 0.01",
            sc_accuracy_rho=1e-5,
            sc_accuracy_eev=1e-3,
            sc_accuracy_etot=1e-6,
            sc_accuracy_forces=1e-4,
            extra_keywords={"mixer": "pulay", "n_max_pulay": "10", "charge_mix_param": "0.5"},
        )
        ctrl = gen.generate(water, params).files["control.in"]
        assert "xc                 pbe" in ctrl
        assert "relativistic       none" in ctrl
        assert "relax_geometry     bfgs 1e-2" in ctrl
        assert "sc_accuracy_forces 1e-04" in ctrl

    def test_gmtkn55_pbe0_xdm_bj_tight(
        self, gen: FHIAimsGenerator, methane: MoleculeSpec
    ) -> None:
        """Matches GMTKN55 PBE0/ACONF convention: xdm bj tight."""
        params = FHIAimsParams(
            xc=XCFunctional.PBE0,
            basis=BasisSetTier.TIGHT,
            dispersion=DispersionMethod.XDM,
            xdm_params=XDMParams(keyword="bj tight"),
            sc_accuracy_rho=1e-7,
        )
        ctrl = gen.generate(methane, params).files["control.in"]
        assert "xc                 pbe0" in ctrl
        assert "xdm bj tight" in ctrl
        assert "sc_accuracy_rho    1e-07" in ctrl


class TestGeometryIn:
    """Tests for geometry.in generation."""

    def test_molecular_geometry(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        geom = gen.generate(water).files["geometry.in"]
        assert "atom" in geom
        assert "atom_frac" not in geom
        lines = [l for l in geom.splitlines() if l.startswith("atom")]
        assert len(lines) == 3
        assert "O" in lines[0]
        assert "H" in lines[1]

    def test_periodic_geometry(
        self, gen: FHIAimsGenerator, nacl_crystal: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(k_grid=[4, 4, 4])
        geom = gen.generate(nacl_crystal, params).files["geometry.in"]
        assert "lattice_vector" in geom
        lattice_lines = [l for l in geom.splitlines() if l.startswith("lattice_vector")]
        assert len(lattice_lines) == 3
        atom_lines = [l for l in geom.splitlines() if l.startswith("atom")]
        assert len(atom_lines) == 8

    def test_diamond_geometry(
        self, gen: FHIAimsGenerator, diamond: MoleculeSpec
    ) -> None:
        """Diamond geometry should have lattice_vector and atom lines."""
        params = FHIAimsParams(k_grid=[2, 2, 2])
        geom = gen.generate(diamond, params).files["geometry.in"]
        lattice_lines = [l for l in geom.splitlines() if l.startswith("lattice_vector")]
        assert len(lattice_lines) == 3
        atom_lines = [l for l in geom.splitlines() if l.startswith("atom")]
        assert len(atom_lines) == 2

    def test_xyz_input(
        self, gen: FHIAimsGenerator, water_xyz: MoleculeSpec
    ) -> None:
        geom = gen.generate(water_xyz).files["geometry.in"]
        atom_lines = [l for l in geom.splitlines() if l.startswith("atom")]
        assert len(atom_lines) == 3


class TestSummary:
    def test_summary_content(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        result = gen.generate(water)
        assert "H2O" in result.summary
        assert "pbe" in result.summary
        assert "light" in result.summary

    def test_summary_periodic(
        self, gen: FHIAimsGenerator, nacl_crystal: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(k_grid=[4, 4, 4])
        result = gen.generate(nacl_crystal, params)
        assert "periodic" in result.summary

    def test_summary_spin(
        self, gen: FHIAimsGenerator, oxygen_triplet: MoleculeSpec
    ) -> None:
        result = gen.generate(oxygen_triplet)
        assert "spin" in result.summary

    def test_summary_xdm(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        params = FHIAimsParams(dispersion=DispersionMethod.XDM)
        result = gen.generate(water, params)
        assert "xdm" in result.summary


class TestGeneratedFiles:
    def test_produces_two_files(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        result = gen.generate(water)
        assert set(result.files.keys()) == {"control.in", "geometry.in"}

    def test_files_end_with_newline(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        result = gen.generate(water)
        for content in result.files.values():
            assert content.endswith("\n")

    def test_no_leading_whitespace_on_keywords(
        self, gen: FHIAimsGenerator, water: MoleculeSpec
    ) -> None:
        """FHI-aims convention: keywords are left-aligned, not indented."""
        ctrl = gen.generate(water).files["control.in"]
        for line in ctrl.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            # Main keywords should not be indented
            assert not line.startswith("  "), f"Unexpected indent: {line!r}"
