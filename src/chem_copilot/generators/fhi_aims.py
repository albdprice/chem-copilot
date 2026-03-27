"""FHI-aims input file generator.

Generates control.in and geometry.in files from a molecule specification
and calculation parameters, following FHI-aims manual conventions.

The output formatting matches FHI-aims regression test conventions:
keywords are left-aligned (no leading whitespace), species blocks use
two-space indent for sub-keywords.

Reference: FHI-aims manual, regression_tests/testcases/
"""

from __future__ import annotations

from chem_copilot.models import (
    BasisSetTier,
    CalculationType,
    DispersionMethod,
    FHIAimsParams,
    GeneratedInput,
    MoleculeSpec,
    RelativisticTreatment,
    XCFunctional,
    XDMParams,
)

# Atomic numbers for species_defaults paths.
_ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86,
}

# FHI-aims XC functional keywords (value after ``xc``).
# Most are plain names; HSE06 needs the omega parameter; meta-GGAs use dfauto.
_XC_KEYWORD: dict[str, str] = {
    XCFunctional.PBE.value: "pbe",
    XCFunctional.PBE0.value: "pbe0",
    XCFunctional.HSE06.value: "hse06  0.11",
    XCFunctional.BLYP.value: "blyp",
    XCFunctional.B3LYP.value: "b3lyp",
    XCFunctional.B86BPBE.value: "b86bpbe",
    XCFunctional.RPBE.value: "rpbe",
    XCFunctional.PW_LDA.value: "pw-lda",
    XCFunctional.SCAN.value: "dfauto scan",
    XCFunctional.TPSS.value: "dfauto tpss",
    XCFunctional.M06_2X.value: "dfauto m06-2x",
    XCFunctional.M06L.value: "dfauto m06l",
    XCFunctional.REVPBE.value: "revpbe",
    XCFunctional.AM05.value: "am05",
    XCFunctional.HF.value: "hf",
}

# Relativistic keyword mapping.
_RELATIVISTIC_KEYWORD: dict[str, str] = {
    RelativisticTreatment.NONE.value: "none",
    RelativisticTreatment.ATOMIC_ZORA.value: "atomic_zora scalar",
    RelativisticTreatment.ZORA.value: "zora scalar 1e-12",
}


class FHIAimsGenerator:
    """Generate FHI-aims control.in and geometry.in files.

    Output formatting follows FHI-aims conventions observed in the
    regression test suite: keywords are left-aligned with no leading
    whitespace (species sub-keywords use two-space indent).
    """

    def generate(
        self,
        molecule: MoleculeSpec,
        params: FHIAimsParams | None = None,
    ) -> GeneratedInput:
        """Generate FHI-aims input files.

        Parameters
        ----------
        molecule : MoleculeSpec
            Molecular or periodic system specification.
        params : FHIAimsParams, optional
            Calculation parameters. Defaults to a single-point PBE/light calculation.

        Returns
        -------
        GeneratedInput
            Generated file contents, warnings, and a human-readable summary.
        """
        if params is None:
            params = FHIAimsParams()

        warnings: list[str] = []

        # Auto-configure spin from multiplicity
        spin_collinear = params.spin == "collinear" or (
            params.spin is None and molecule.multiplicity > 1
        )
        if molecule.multiplicity > 1 and params.spin is None:
            warnings.append(
                f"Multiplicity {molecule.multiplicity} requires spin-polarized "
                "calculation. Setting spin collinear automatically."
            )

        # Periodic system checks
        if molecule.is_periodic and params.k_grid is None:
            warnings.append(
                "Periodic system detected but no k_grid specified. "
                "Defaulting to Gamma-point [1, 1, 1]. "
                "Consider a denser grid for production calculations."
            )
            params = params.model_copy(update={"k_grid": [1, 1, 1]})

        if not molecule.is_periodic and params.k_grid is not None:
            warnings.append("k_grid specified for a non-periodic system — ignoring.")

        control_in = self._build_control_in(molecule, params, spin_collinear, warnings)
        geometry_in = self._build_geometry_in(molecule)

        return GeneratedInput(
            files={"control.in": control_in, "geometry.in": geometry_in},
            warnings=warnings,
            summary=self._build_summary(molecule, params, spin_collinear),
        )

    # ------------------------------------------------------------------
    # control.in
    # ------------------------------------------------------------------

    def _build_control_in(
        self,
        molecule: MoleculeSpec,
        params: FHIAimsParams,
        spin_collinear: bool,
        warnings: list[str],
    ) -> str:
        sections: list[str] = []

        # Header
        sections.append(
            "#" + "=" * 70 + "\n"
            "#\n"
            "#  FHI-aims control.in — generated by chem-copilot\n"
            "#\n"
            "#" + "=" * 70
        )

        # --- Physical model ------------------------------------------------
        block: list[str] = ["#", "#  Physical model", "#"]
        xc_kw = _XC_KEYWORD.get(params.xc.value, params.xc.value)
        block.append(f"xc                 {xc_kw}")

        if params.xc == XCFunctional.HSE06:
            block.append("hse_unit bohr-1")

        # xc_pre — preconverge with cheaper functional
        if params.xc_pre is not None:
            func, n_steps = params.xc_pre
            block.append(f"xc_pre             {func} {n_steps}")

        # Spin
        if spin_collinear:
            block.append("spin               collinear")
            block.append(f"default_initial_moment {molecule.n_unpaired_electrons}")
        else:
            block.append("spin               none")

        # Relativistic
        rel_kw = _RELATIVISTIC_KEYWORD.get(
            params.relativistic.value, "atomic_zora scalar"
        )
        block.append(f"relativistic       {rel_kw}")

        # Charge
        if molecule.charge != 0:
            block.append(f"charge             {molecule.charge}")

        # Dispersion
        self._add_dispersion(block, params, warnings)

        # Compute forces / stress
        if params.compute_forces:
            block.append("compute_forces .true.")
        if params.compute_analytical_stress and molecule.is_periodic:
            block.append("compute_analytical_stress .true.")

        sections.append("\n".join(block))

        # --- SCF convergence -----------------------------------------------
        block = [
            "#",
            "#  SCF convergence",
            "#",
            f"occupation_type    {params.occupation_type}",
            f"sc_accuracy_rho    {params.sc_accuracy_rho:.0e}",
            f"sc_accuracy_eev    {params.sc_accuracy_eev:.0e}",
            f"sc_accuracy_etot   {params.sc_accuracy_etot:.0e}",
        ]
        needs_forces = params.calculation_type in (
            CalculationType.OPTIMIZATION,
            CalculationType.MOLECULAR_DYNAMICS,
        ) or params.compute_forces
        if params.sc_accuracy_forces is not None:
            block.append(f"sc_accuracy_forces {params.sc_accuracy_forces:.0e}")
        elif needs_forces:
            block.append("sc_accuracy_forces 1e-4")
        sections.append("\n".join(block))

        # --- k-grid (periodic) ---------------------------------------------
        if molecule.is_periodic and params.k_grid:
            k = params.k_grid
            sections.append(f"k_grid             {k[0]} {k[1]} {k[2]}")

        # --- Calculation-type specifics ------------------------------------
        if params.calculation_type == CalculationType.OPTIMIZATION:
            block = [
                "#",
                "#  Geometry relaxation",
                "#",
                f"relax_geometry     {params.relax_geometry}",
            ]
            if molecule.is_periodic:
                block.append("relax_unit_cell    full")
            sections.append("\n".join(block))

        elif params.calculation_type == CalculationType.MOLECULAR_DYNAMICS:
            ensemble = params.md_ensemble or "NVT_berendsen"
            block = [
                "#",
                "#  Molecular dynamics",
                "#",
                f"MD_run             {ensemble}",
            ]
            if params.md_temperature is not None:
                block.append(f"MD_temperature     {params.md_temperature}")
            if params.md_time_step is not None:
                block.append(f"MD_time_step       {params.md_time_step}")
            if params.md_n_steps is not None:
                block.append("MD_schedule")
                block.append(f"  MD_segment       {params.md_n_steps}")
                block.append("MD_schedule_end")
            sections.append("\n".join(block))

        # --- Output requests -----------------------------------------------
        if params.output_requests:
            block = ["#", "#  Output", "#"]
            for req in params.output_requests:
                block.append(f"output             {req}")
            sections.append("\n".join(block))

        # --- Extra keywords ------------------------------------------------
        if params.extra_keywords:
            block = ["#", "#  Additional settings", "#"]
            for key, value in params.extra_keywords.items():
                block.append(f"{key:<18s} {value}")
            sections.append("\n".join(block))

        # --- Species defaults (commented include directives) ---------------
        unique_elements = _unique_elements(molecule)
        block = [
            "#",
            "#" + "-" * 70,
            f"#  Basis set: {params.basis.value}",
            "#  Append species defaults below or set species_dir.",
            "#" + "-" * 70,
        ]
        for elem in unique_elements:
            path = _species_defaults_path(params.basis, elem)
            block.append(f"#  {path}")
        sections.append("\n".join(block))

        return "\n\n".join(sections) + "\n"

    def _add_dispersion(
        self,
        block: list[str],
        params: FHIAimsParams,
        warnings: list[str],
    ) -> None:
        """Append dispersion keywords to the block."""
        if params.dispersion == DispersionMethod.NONE:
            return

        if params.dispersion == DispersionMethod.XDM:
            xdm = params.xdm_params
            if xdm and xdm.a1 is not None and xdm.a2 is not None:
                # Explicit damping: xdm <a1> <a2>
                block.append(f"xdm {xdm.a1:.8f} {xdm.a2:.8f}")
            elif xdm and xdm.keyword:
                # Keyword form: xdm bj tight, xdm lightdense, etc.
                block.append(f"xdm {xdm.keyword}")
            else:
                # Default: xdm bj <basis_tier>
                block.append(f"xdm bj {params.basis.value}")

        elif params.dispersion == DispersionMethod.TS:
            block.append("vdw_correction_hirshfeld")

        elif params.dispersion in (DispersionMethod.MBD, DispersionMethod.MBD_NL):
            if params.dispersion == DispersionMethod.MBD_NL:
                block.append("many_body_dispersion_nl")
            else:
                block.append("many_body_dispersion")

        elif params.dispersion == DispersionMethod.TS_SCS:
            block.append("many_body_dispersion_nl")

        else:
            warnings.append(
                f"Dispersion method '{params.dispersion.value}' is not natively "
                "supported in FHI-aims. For D3/D4 use Psi4 or an external library."
            )

    # ------------------------------------------------------------------
    # geometry.in
    # ------------------------------------------------------------------

    def _build_geometry_in(self, molecule: MoleculeSpec) -> str:
        lines: list[str] = [
            "#" + "=" * 70,
            "#  FHI-aims geometry.in",
            "#  Generated by chem-copilot",
            "#" + "=" * 70,
        ]

        # Lattice vectors (periodic)
        if molecule.is_periodic and molecule.lattice_vectors:
            for vec in molecule.lattice_vectors:
                lines.append(
                    f"lattice_vector {vec[0]:16.10f} {vec[1]:16.10f} {vec[2]:16.10f}"
                )

        # Atom positions — always Cartesian ``atom`` since MoleculeSpec stores Cartesian.
        symbols, coords = molecule.to_symbols_coords()
        for sym, xyz in zip(symbols, coords):
            lines.append(
                f"atom {xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f} {sym}"
            )

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        molecule: MoleculeSpec,
        params: FHIAimsParams,
        spin_collinear: bool,
    ) -> str:
        symbols, _ = molecule.to_symbols_coords()
        formula = _hill_formula(symbols)
        parts = [
            f"FHI-aims {params.calculation_type.value}",
            f"{formula} ({len(symbols)} atoms)",
        ]
        if molecule.is_periodic:
            parts.append(f"periodic k={params.k_grid}")
        parts.append(f"{params.xc.value}/{params.basis.value}")
        if params.dispersion != DispersionMethod.NONE:
            parts.append(f"+{params.dispersion.value}")
        if spin_collinear:
            parts.append(f"spin (mult={molecule.multiplicity})")
        if molecule.charge != 0:
            parts.append(f"charge={molecule.charge}")
        return " | ".join(parts)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _unique_elements(molecule: MoleculeSpec) -> list[str]:
    """Return unique element symbols in order of first appearance."""
    symbols, _ = molecule.to_symbols_coords()
    seen: set[str] = set()
    result: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _species_defaults_path(basis: BasisSetTier, element: str) -> str:
    """Standard FHI-aims species_defaults path for an element.

    Convention: defaults_2020/<tier>/<ZZ>_<Element>_default
    e.g. defaults_2020/tight/06_C_default
    """
    tier = basis.value
    z = _ATOMIC_NUMBERS.get(element, 0)
    return f"defaults_2020/{tier}/{z:02d}_{element}_default"


def _hill_formula(symbols: list[str]) -> str:
    """Molecular formula in Hill system (C first, H second, then alphabetical)."""
    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    parts: list[str] = []
    for elem in ["C", "H"]:
        if elem in counts:
            n = counts.pop(elem)
            parts.append(f"{elem}{n}" if n > 1 else elem)
    for elem in sorted(counts):
        n = counts[elem]
        parts.append(f"{elem}{n}" if n > 1 else elem)
    return "".join(parts)
