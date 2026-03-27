"""Output file parser using cclib + custom FHI-aims parsing.

Parses electronic structure calculation output files from FHI-aims,
Psi4, and Gaussian, returning structured Pydantic models.

cclib does not have an FHI-aims parser, so we provide one.
cclib auto-detection fails for Psi4, so we use explicit parser selection.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from chem_copilot.models import (
    CalculationResult,
    OptimizationTrajectory,
    ParsedEnergies,
    ParsedForces,
    ParsedFrequencies,
    ParsedOrbitals,
)

logger = logging.getLogger(__name__)

# Conversion factors
EV_PER_HARTREE = 27.211386245988
ANGSTROM_PER_BOHR = 0.529177210903


class OutputParser:
    """Parse computational chemistry output files into structured results."""

    def parse_file(
        self,
        filepath: str | Path,
        program: str | None = None,
    ) -> CalculationResult:
        """Parse an output file and return a structured result.

        Parameters
        ----------
        filepath : str or Path
            Path to the output file.
        program : str, optional
            Program hint: 'fhi-aims', 'psi4', 'gaussian'.
            If None, auto-detect from file content.

        Returns
        -------
        CalculationResult
            Structured parsed data.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Output file not found: {filepath}")

        if program is None:
            program = self._detect_program_from_file(filepath)

        if program == "fhi-aims":
            return self._parse_fhiaims(filepath)

        # Use cclib for Psi4 and Gaussian
        data = self._cclib_parse(filepath, program)
        if data is None:
            raise ValueError(
                f"Could not parse {filepath}. "
                "Try specifying the program explicitly."
            )

        detected = self._detect_program_from_cclib(data, filepath, program)
        return self._build_result(data, detected)

    def parse_string(
        self,
        content: str,
        program: str = "unknown",
    ) -> CalculationResult:
        """Parse output content from a string."""
        import tempfile

        suffix = {"fhi-aims": ".out", "psi4": ".out", "gaussian": ".log"}.get(
            program.lower(), ".out"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False
        ) as f:
            f.write(content)
            f.flush()
            return self.parse_file(f.name, program=program)

    # ------------------------------------------------------------------
    # cclib-based parsing (Psi4, Gaussian)
    # ------------------------------------------------------------------

    @staticmethod
    def _cclib_parse(filepath: Path, program: str) -> Any:
        """Parse with cclib, using explicit parser when auto-detect fails."""
        import cclib

        # Try auto-detect first
        data = cclib.io.ccread(str(filepath))
        if data is not None:
            return data

        # Auto-detect failed — try explicit parser
        parser_map = {
            "psi4": cclib.parser.Psi4,
            "gaussian": cclib.parser.Gaussian,
        }
        parser_cls = parser_map.get(program.lower())
        if parser_cls is not None:
            try:
                return parser_cls(str(filepath)).parse()
            except Exception as exc:
                logger.warning("cclib %s parser failed: %s", program, exc)

        return None

    # ------------------------------------------------------------------
    # FHI-aims custom parser
    # ------------------------------------------------------------------

    def _parse_fhiaims(self, filepath: Path) -> CalculationResult:
        """Parse an FHI-aims output file directly (cclib has no FHI-aims support)."""
        text = filepath.read_text()
        warnings: list[str] = []

        energies = self._fhiaims_energies(text)
        forces = self._fhiaims_forces(text)
        orbitals = self._fhiaims_orbitals(text)
        optimization = self._fhiaims_optimization(text)
        n_atoms = self._fhiaims_natom(text)
        success = self._fhiaims_success(text)

        calc_type = "single_point"
        if optimization.n_steps > 1:
            calc_type = "optimization"

        return CalculationResult(
            program="fhi-aims",
            calc_type=calc_type,
            success=success,
            energies=energies,
            forces=forces,
            orbitals=orbitals,
            frequencies=ParsedFrequencies(),
            optimization=optimization,
            n_atoms=n_atoms,
            n_basis_functions=self._fhiaims_nbasis(text),
            warnings=warnings,
            raw_metadata=self._fhiaims_metadata(text),
        )

    @staticmethod
    def _fhiaims_energies(text: str) -> ParsedEnergies:
        """Extract energies from FHI-aims output.

        FHI-aims uses Fortran-style scientific notation in the detailed output:
          | Total energy corrected        :         -0.787841351866946E+02 eV
        but plain decimals in the summary line:
          | Total energy of the DFT / Hartree-Fock s.c.f. calculation      :            -78.784135187 eV
        We try the summary line first, then fall back to the detailed form.
        """
        # Regex for a floating-point number including Fortran-style 0.123E+02
        _FLOAT = r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?"

        scf_energies: list[float] = []
        total_energy: float | None = None
        total_energy_ha: float | None = None

        # Primary: summary line (plain decimal, most readable)
        # "| Total energy of the DFT / Hartree-Fock s.c.f. calculation      :  -78.784135187 eV"
        for m in re.finditer(
            r"\| Total energy of the DFT / Hartree-Fock s\.c\.f\. calculation\s*:\s*(" + _FLOAT + r")\s*eV",
            text,
        ):
            scf_energies.append(float(m.group(1)))

        # Fallback: detailed corrected energy (Fortran notation)
        if not scf_energies:
            for m in re.finditer(
                r"\| Total energy corrected\s*:\s*(" + _FLOAT + r")\s*eV",
                text,
            ):
                scf_energies.append(float(m.group(1)))

        # Fallback: uncorrected
        if not scf_energies:
            for m in re.finditer(
                r"\| Total energy uncorrected\s*:\s*(" + _FLOAT + r")\s*eV",
                text,
            ):
                scf_energies.append(float(m.group(1)))

        if scf_energies:
            total_energy = scf_energies[-1]
            total_energy_ha = total_energy / EV_PER_HARTREE

        # XDM / dispersion energy
        disp_energy: float | None = None
        m = re.search(r"XDM dispersion energy\s*:\s*(" + _FLOAT + r")\s*Ha", text)
        if m:
            disp_energy = float(m.group(1))

        return ParsedEnergies(
            total_energy=total_energy,
            total_energy_hartree=total_energy_ha,
            scf_energies=scf_energies,
            dispersion_energy=disp_energy,
        )

    @staticmethod
    def _fhiaims_forces(text: str) -> ParsedForces:
        """Extract forces from FHI-aims output.

        Format:
          Total atomic forces (unitary forces cleaned) [eV/Ang]:
          |    1         -0.174242605163267E-12         -0.268176151466238E+01          0.811297123282653E-28
        """
        _FLOAT = r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?"
        forces_blocks = re.findall(
            r"Total atomic forces.*?\[eV/Ang\]:\s*\n((?:\s*\|.*\n)+)",
            text,
        )
        if not forces_blocks:
            return ParsedForces()

        last_block = forces_blocks[-1]
        forces: list[list[float]] = []
        for line in last_block.strip().splitlines():
            # Strip the leading "  |" and parse remaining numbers
            clean = line.replace("|", "").strip()
            nums = re.findall(_FLOAT, clean)
            if len(nums) >= 4:
                # nums[0] = atom index, nums[1:4] = fx, fy, fz
                forces.append([float(nums[1]), float(nums[2]), float(nums[3])])

        if not forces:
            return ParsedForces()

        forces_arr = np.array(forces)
        return ParsedForces(
            forces=forces,
            max_force=float(np.max(np.abs(forces_arr))),
            rms_force=float(np.sqrt(np.mean(forces_arr**2))),
        )

    @staticmethod
    def _fhiaims_orbitals(text: str) -> ParsedOrbitals:
        """Extract orbital eigenvalues from FHI-aims output."""
        # "Writing Kohn-Sham eigenvalues."
        # Then lines like "   1      2      -19.16472189     -0.70384218"
        # Format: state  occ  eigenvalue(eV)  eigenvalue(Ha)
        eigenvalues: list[float] = []
        occupations: list[float] = []

        # Find the last eigenvalue block
        blocks = re.findall(
            r"Writing Kohn-Sham eigenvalues\.\s*\n((?:.*\n)*?)(?:\n\s*\n|\Z)",
            text,
        )
        if not blocks:
            return ParsedOrbitals()

        last_block = blocks[-1]
        for line in last_block.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                try:
                    _state = int(parts[0])
                    occ = float(parts[1])
                    ev = float(parts[2])
                    eigenvalues.append(ev)
                    occupations.append(occ)
                except ValueError:
                    continue

        if not eigenvalues:
            return ParsedOrbitals()

        # Find HOMO/LUMO
        homo_idx = None
        for i, occ in enumerate(occupations):
            if occ > 0.5:
                homo_idx = i

        homo_e = eigenvalues[homo_idx] if homo_idx is not None else None
        lumo_e = None
        gap = None
        if homo_idx is not None and homo_idx + 1 < len(eigenvalues):
            lumo_e = eigenvalues[homo_idx + 1]
            if homo_e is not None:
                gap = lumo_e - homo_e

        return ParsedOrbitals(
            eigenvalues=eigenvalues,
            occupations=occupations,
            homo_index=homo_idx,
            homo_energy=homo_e,
            lumo_energy=lumo_e,
            homo_lumo_gap=gap,
        )

    @staticmethod
    def _fhiaims_optimization(text: str) -> OptimizationTrajectory:
        """Extract optimization trajectory from FHI-aims output.

        Uses "Total energy corrected" (Fortran notation, appears every SCF)
        and "Maximum force component is" (appears every geometry step).
        The number of force lines determines the number of geometry steps.
        """
        _FLOAT = r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?"

        step_forces: list[float] = []
        for m in re.finditer(
            r"Maximum force component is\s+(" + _FLOAT + r")", text
        ):
            step_forces.append(float(m.group(1)))

        if not step_forces:
            return OptimizationTrajectory()

        # Collect corrected energies — one per SCF convergence per geometry step
        all_energies: list[float] = []
        for m in re.finditer(
            r"\| Total energy corrected\s*:\s*(" + _FLOAT + r")\s*eV",
            text,
        ):
            all_energies.append(float(m.group(1)))

        # Fall back to summary line if corrected not found
        if not all_energies:
            for m in re.finditer(
                r"\| Total energy of the DFT / Hartree-Fock s\.c\.f\. calculation\s*:\s*(" + _FLOAT + r")\s*eV",
                text,
            ):
                all_energies.append(float(m.group(1)))

        # Match energy count to force count (take last N energies)
        n_steps = len(step_forces)
        if len(all_energies) >= n_steps:
            step_energies = all_energies[-n_steps:]
        else:
            step_energies = all_energies

        converged = "Final atomic structure" in text

        return OptimizationTrajectory(
            energies=step_energies,
            max_forces=step_forces,
            converged=converged,
            n_steps=n_steps,
        )

    @staticmethod
    def _fhiaims_natom(text: str) -> int | None:
        m = re.search(r"\| Number of atoms\s*:\s*(\d+)", text)
        return int(m.group(1)) if m else None

    @staticmethod
    def _fhiaims_nbasis(text: str) -> int | None:
        m = re.search(r"\| Maximum number of basis functions\s*:\s*(\d+)", text)
        if m:
            return int(m.group(1))
        m = re.search(r"\| Number of basis functions\s*:\s*(\d+)", text)
        return int(m.group(1)) if m else None

    @staticmethod
    def _fhiaims_success(text: str) -> bool:
        return "Have a nice day." in text

    @staticmethod
    def _fhiaims_metadata(text: str) -> dict[str, object]:
        meta: dict[str, object] = {"package": "FHI-aims"}
        m = re.search(r"FHI-aims version\s*:\s*(.+)", text)
        if m:
            meta["version"] = m.group(1).strip()
        m = re.search(r"Using\s+(\d+)\s+parallel tasks", text)
        if m:
            meta["n_tasks"] = int(m.group(1))
        return meta

    # ------------------------------------------------------------------
    # cclib result builder (Psi4, Gaussian)
    # ------------------------------------------------------------------

    def _build_result(self, data: Any, program: str) -> CalculationResult:
        """Convert cclib parsed data into our Pydantic models."""
        warnings: list[str] = []

        energies = self._extract_energies(data, warnings)
        forces = self._extract_forces(data)
        orbitals = self._extract_orbitals(data)
        frequencies = self._extract_frequencies(data)
        optimization = self._extract_optimization(data)

        calc_type = self._detect_calc_type(data, optimization, frequencies)

        n_atoms = getattr(data, "natom", None)
        n_basis = getattr(data, "nbasis", None)

        success = self._check_success(data, optimization, energies, program)

        return CalculationResult(
            program=program,
            calc_type=calc_type,
            success=success,
            energies=energies,
            forces=forces,
            orbitals=orbitals,
            frequencies=frequencies,
            optimization=optimization,
            n_atoms=n_atoms,
            n_basis_functions=n_basis,
            warnings=warnings,
            raw_metadata=self._raw_metadata(data),
        )

    def _extract_energies(
        self, data: Any, warnings: list[str]
    ) -> ParsedEnergies:
        result: dict[str, Any] = {}

        if hasattr(data, "scfenergies"):
            scf = data.scfenergies.tolist()
            result["scf_energies"] = scf
            result["total_energy"] = scf[-1] if scf else None
            if result["total_energy"] is not None:
                result["total_energy_hartree"] = result["total_energy"] / EV_PER_HARTREE

        if hasattr(data, "ccenergies"):
            cc = data.ccenergies.tolist()
            if cc:
                last_cc_ev = cc[-1]
                result["total_energy"] = last_cc_ev
                result["total_energy_hartree"] = last_cc_ev / EV_PER_HARTREE
                result["ccsd_t_energy"] = last_cc_ev / EV_PER_HARTREE

        if hasattr(data, "mpenergies"):
            mp = data.mpenergies.tolist()
            if mp:
                if isinstance(mp[-1], list):
                    mp2_ev = mp[-1][-1]
                else:
                    mp2_ev = mp[-1]
                result["mp2_energy"] = mp2_ev / EV_PER_HARTREE

        if hasattr(data, "zpve"):
            result["zero_point_energy"] = data.zpve

        return ParsedEnergies(**result)

    def _extract_forces(self, data: Any) -> ParsedForces:
        if not hasattr(data, "grads"):
            return ParsedForces()
        grads = data.grads
        if grads is None or len(grads) == 0:
            return ParsedForces()

        last_grad = grads[-1]
        forces = -last_grad * EV_PER_HARTREE / ANGSTROM_PER_BOHR
        return ParsedForces(
            forces=forces.tolist(),
            max_force=float(np.max(np.abs(forces))),
            rms_force=float(np.sqrt(np.mean(forces**2))),
        )

    def _extract_orbitals(self, data: Any) -> ParsedOrbitals:
        if not hasattr(data, "moenergies"):
            return ParsedOrbitals()

        mo = data.moenergies[0]
        eigenvalues = mo.tolist()

        occupations: list[float] = []
        if hasattr(data, "mooccnos"):
            occupations = data.mooccnos[0].tolist()

        homo_idx = None
        homo_e = None
        lumo_e = None
        gap = None
        if hasattr(data, "homos"):
            homo_idx = int(data.homos[0])
            homo_e = float(mo[homo_idx])
            if homo_idx + 1 < len(mo):
                lumo_e = float(mo[homo_idx + 1])
                gap = lumo_e - homo_e

        return ParsedOrbitals(
            eigenvalues=eigenvalues,
            occupations=occupations,
            homo_index=homo_idx,
            homo_energy=homo_e,
            lumo_energy=lumo_e,
            homo_lumo_gap=gap,
        )

    def _extract_frequencies(self, data: Any) -> ParsedFrequencies:
        if not hasattr(data, "vibfreqs"):
            return ParsedFrequencies()
        freqs = data.vibfreqs.tolist()
        intensities = data.vibirs.tolist() if hasattr(data, "vibirs") else []
        is_imaginary = [f < 0 for f in freqs]
        return ParsedFrequencies(
            frequencies=freqs,
            intensities=intensities,
            is_imaginary=is_imaginary,
            n_imaginary=sum(is_imaginary),
        )

    def _extract_optimization(self, data: Any) -> OptimizationTrajectory:
        if not hasattr(data, "optstatus"):
            if hasattr(data, "scfenergies") and len(data.scfenergies) > 1:
                if hasattr(data, "atomcoords") and len(data.atomcoords) > 1:
                    energies = data.scfenergies.tolist()
                    max_forces: list[float] = []
                    if hasattr(data, "grads") and data.grads is not None:
                        for grad in data.grads:
                            f = -grad * EV_PER_HARTREE / ANGSTROM_PER_BOHR
                            max_forces.append(float(np.max(np.abs(f))))
                    return OptimizationTrajectory(
                        energies=energies, max_forces=max_forces,
                        converged=False, n_steps=len(energies),
                    )
            return OptimizationTrajectory()

        energies = data.scfenergies.tolist() if hasattr(data, "scfenergies") else []
        converged = bool(np.any(data.optstatus == 4)) if len(data.optstatus) > 0 else False

        max_forces: list[float] = []
        if hasattr(data, "grads") and data.grads is not None:
            for grad in data.grads:
                f = -grad * EV_PER_HARTREE / ANGSTROM_PER_BOHR
                max_forces.append(float(np.max(np.abs(f))))

        return OptimizationTrajectory(
            energies=energies, max_forces=max_forces,
            converged=converged, n_steps=len(energies),
        )

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_program_from_file(filepath: Path) -> str:
        """Detect program from file content."""
        try:
            text = filepath.read_text(errors="replace")
        except Exception:
            return "unknown"

        # Check a generous portion — some programs print banners late
        head = text[:20000]

        if "FHI-aims" in head or "Fritz Haber Institute" in head:
            return "fhi-aims"
        if "Psi4" in head or "An Open-Source Ab Initio" in head or "tstart() called on" in head:
            return "psi4"
        if "Gaussian, Inc." in head or "Entering Gaussian System" in head:
            return "gaussian"

        # Fallback: check the whole file for unmistakable signatures
        if "Psi4 exiting successfully" in text or "*** Psi4" in text:
            return "psi4"
        if "Have a nice day." in text and "Total energy of the DFT" in text:
            return "fhi-aims"

        return "unknown"

    @staticmethod
    def _detect_program_from_cclib(
        data: Any, filepath: Path, hint: str,
    ) -> str:
        metadata = getattr(data, "metadata", {})
        if "package" in metadata:
            pkg = metadata["package"].lower()
            if "psi" in pkg:
                return "psi4"
            if "gaussian" in pkg:
                return "gaussian"
            return pkg
        return hint

    @staticmethod
    def _detect_calc_type(
        data: Any,
        optimization: OptimizationTrajectory,
        frequencies: ParsedFrequencies,
    ) -> str:
        if frequencies.frequencies:
            return "frequency"
        if optimization.n_steps > 1:
            return "optimization"
        return "single_point"

    @staticmethod
    def _check_success(
        data: Any,
        optimization: OptimizationTrajectory,
        energies: ParsedEnergies,
        program: str,
    ) -> bool:
        """Check if calculation completed normally.

        Note: cclib marks Psi4 success=False even for completed calculations,
        so we don't trust it blindly for Psi4.
        """
        metadata = getattr(data, "metadata", {})

        # Gaussian metadata.success is reliable
        if program == "gaussian" and "success" in metadata:
            return bool(metadata["success"])

        # For Psi4 and others, use energy-based heuristic
        if energies.total_energy is not None:
            if optimization.n_steps > 1:
                return optimization.converged
            return True

        return False

    @staticmethod
    def _raw_metadata(data: Any) -> dict[str, object]:
        meta: dict[str, object] = {}
        for attr in ["metadata", "charge", "mult"]:
            if hasattr(data, attr):
                val = getattr(data, attr)
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                meta[attr] = val
        return meta
