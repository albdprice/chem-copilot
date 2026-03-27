"""Pydantic models for structured input/output across all tools."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CalculationType(str, Enum):
    """Supported calculation types."""
    SINGLE_POINT = "single_point"
    OPTIMIZATION = "optimization"
    MOLECULAR_DYNAMICS = "molecular_dynamics"
    FREQUENCY = "frequency"
    BAND_STRUCTURE = "band_structure"


class XCFunctional(str, Enum):
    """Exchange-correlation functionals."""
    PBE = "pbe"
    PBE0 = "pbe0"
    HSE06 = "hse06"
    BLYP = "blyp"
    B3LYP = "b3lyp"
    B86BPBE = "b86bpbe"
    RPBE = "rpbe"
    PW_LDA = "pw-lda"
    SCAN = "scan"
    TPSS = "tpss"
    M06_2X = "m06-2x"
    M06L = "m06l"
    REVPBE = "revpbe"
    AM05 = "am05"
    HF = "hf"


class BasisSetTier(str, Enum):
    """FHI-aims NAO basis set tiers."""
    LIGHT = "light"
    INTERMEDIATE = "intermediate"
    TIGHT = "tight"
    REALLY_TIGHT = "really_tight"


class DispersionMethod(str, Enum):
    """Dispersion correction methods."""
    NONE = "none"
    TS = "tkatchenko-scheffler"
    TS_SCS = "ts-scs"
    MBD = "mbd"
    MBD_NL = "mbd-nl"
    XDM = "xdm"
    D3 = "d3"
    D3BJ = "d3bj"
    D4 = "d4"


class Psi4Method(str, Enum):
    """Psi4 electronic structure methods."""
    HF = "hf"
    DFT = "dft"
    MP2 = "mp2"
    MP3 = "mp3"
    CCSD = "ccsd"
    CCSD_T = "ccsd(t)"
    SAPT0 = "sapt0"
    ADC2 = "adc(2)"


class RelativisticTreatment(str, Enum):
    """Relativistic treatment options for FHI-aims."""
    NONE = "none"
    ATOMIC_ZORA = "atomic_zora"
    ZORA = "zora"


# ---------------------------------------------------------------------------
# Molecule specification
# ---------------------------------------------------------------------------

class MoleculeSpec(BaseModel):
    """Specification for a molecular or periodic system.

    Accepts either an XYZ string, a list of symbols + coordinates,
    or references an ASE Atoms object (passed at runtime).
    """
    symbols: list[str] = Field(default_factory=list, description="Element symbols")
    coords: list[list[float]] = Field(
        default_factory=list,
        description="Cartesian coordinates in Angstrom, shape (N, 3)",
    )
    xyz_string: Optional[str] = Field(
        default=None,
        description="Raw XYZ-format string (takes precedence over symbols/coords)",
    )
    charge: int = Field(default=0, description="Net charge")
    multiplicity: int = Field(default=1, description="Spin multiplicity (2S+1)")
    lattice_vectors: Optional[list[list[float]]] = Field(
        default=None,
        description="Lattice vectors for periodic systems, shape (3, 3) in Angstrom",
    )

    @field_validator("coords")
    @classmethod
    def validate_coords(cls, v: list[list[float]]) -> list[list[float]]:
        for row in v:
            if len(row) != 3:
                raise ValueError(f"Each coordinate must have 3 components, got {len(row)}")
        return v

    @field_validator("lattice_vectors")
    @classmethod
    def validate_lattice(
        cls, v: Optional[list[list[float]]]
    ) -> Optional[list[list[float]]]:
        if v is not None:
            if len(v) != 3:
                raise ValueError(f"Need 3 lattice vectors, got {len(v)}")
            for row in v:
                if len(row) != 3:
                    raise ValueError("Each lattice vector must have 3 components")
        return v

    @model_validator(mode="after")
    def check_consistency(self) -> MoleculeSpec:
        if self.xyz_string is None and self.symbols:
            if len(self.symbols) != len(self.coords):
                raise ValueError(
                    f"symbols ({len(self.symbols)}) and coords ({len(self.coords)}) "
                    "must have the same length"
                )
        return self

    @property
    def is_periodic(self) -> bool:
        return self.lattice_vectors is not None

    @property
    def n_atoms(self) -> int:
        if self.xyz_string:
            return self._count_atoms_xyz()
        return len(self.symbols)

    @property
    def n_unpaired_electrons(self) -> int:
        return self.multiplicity - 1

    def _count_atoms_xyz(self) -> int:
        lines = [l.strip() for l in (self.xyz_string or "").strip().splitlines() if l.strip()]
        if not lines:
            return 0
        # Standard XYZ: first line is atom count
        try:
            return int(lines[0])
        except ValueError:
            # Bare XYZ without header — count data lines
            return sum(1 for l in lines if len(l.split()) >= 4)

    def to_symbols_coords(self) -> tuple[list[str], list[list[float]]]:
        """Return (symbols, coords) regardless of input format."""
        if self.xyz_string:
            return self._parse_xyz()
        return self.symbols, self.coords

    def _parse_xyz(self) -> tuple[list[str], list[list[float]]]:
        lines = [l.strip() for l in (self.xyz_string or "").strip().splitlines() if l.strip()]
        symbols: list[str] = []
        coords: list[list[float]] = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    symbols.append(parts[0])
                    coords.append([x, y, z])
                except ValueError:
                    continue
        return symbols, coords


# ---------------------------------------------------------------------------
# FHI-aims parameters
# ---------------------------------------------------------------------------

class XDMParams(BaseModel):
    """XDM dispersion correction parameters for FHI-aims.

    FHI-aims supports two XDM invocation forms:
      - Explicit damping: ``xdm <a1> <a2>`` (e.g., ``xdm 0.6513 1.4735``)
      - Keyword shorthand: ``xdm bj <tier>`` (e.g., ``xdm bj tight``)
        or ``xdm <tier>`` (e.g., ``xdm lightdense``)

    When ``a1`` and ``a2`` are provided, they take precedence.
    Otherwise ``keyword`` is used (default: ``bj <basis_tier>``).
    """
    a1: Optional[float] = Field(default=None, description="XDM a1 damping parameter")
    a2: Optional[float] = Field(default=None, description="XDM a2 damping parameter (Angstrom)")
    keyword: Optional[str] = Field(
        default=None,
        description="XDM keyword form, e.g. 'bj tight', 'lightdense'. "
        "If None, defaults to 'bj <basis_tier>' at generation time.",
    )

    @model_validator(mode="after")
    def check_a1_a2_pair(self) -> XDMParams:
        if (self.a1 is None) != (self.a2 is None):
            raise ValueError("a1 and a2 must both be provided or both omitted")
        return self


class FHIAimsParams(BaseModel):
    """Parameters for an FHI-aims calculation."""
    xc: XCFunctional = Field(default=XCFunctional.PBE, description="XC functional")
    basis: BasisSetTier = Field(default=BasisSetTier.LIGHT, description="NAO basis tier")
    calculation_type: CalculationType = Field(default=CalculationType.SINGLE_POINT)
    dispersion: DispersionMethod = Field(default=DispersionMethod.NONE)
    xdm_params: Optional[XDMParams] = Field(
        default=None,
        description="XDM-specific parameters (a1/a2 or keyword). "
        "Only used when dispersion=xdm.",
    )
    relativistic: RelativisticTreatment = Field(default=RelativisticTreatment.ATOMIC_ZORA)
    spin: Optional[str] = Field(
        default=None,
        description="Spin treatment: 'collinear' or 'non-collinear'. "
        "Auto-set from multiplicity if None.",
    )
    k_grid: Optional[list[int]] = Field(
        default=None,
        description="k-point grid [k1, k2, k3] for periodic calculations",
    )
    occupation_type: str = Field(
        default="gaussian 0.01",
        description="Occupation smearing type and width (eV)",
    )
    sc_accuracy_rho: float = Field(default=1e-6, description="SCF density convergence")
    sc_accuracy_eev: float = Field(default=1e-4, description="SCF eigenvalue convergence (eV)")
    sc_accuracy_etot: float = Field(default=1e-6, description="SCF total energy convergence (eV)")
    sc_accuracy_forces: Optional[float] = Field(
        default=None,
        description="SCF force convergence (eV/Angstrom)",
    )
    compute_forces: bool = Field(
        default=False,
        description="Explicitly request force computation (.true.)",
    )
    compute_analytical_stress: bool = Field(
        default=False,
        description="Compute analytical stress tensor for periodic systems",
    )
    xc_pre: Optional[tuple[str, int]] = Field(
        default=None,
        description="Preconverge with a cheaper functional, e.g. ('pbe', 10) for 10 SCF steps",
    )
    relax_geometry: str = Field(
        default="trm 1e-3",
        description="Relaxation algorithm and force criterion",
    )
    md_ensemble: Optional[str] = Field(
        default=None,
        description="MD ensemble: NVE, NVT_berendsen, NVT_nose-hoover, etc.",
    )
    md_temperature: Optional[float] = Field(default=None, description="MD temperature (K)")
    md_time_step: Optional[float] = Field(default=None, description="MD time step (fs)")
    md_n_steps: Optional[int] = Field(default=None, description="Number of MD steps")
    output_requests: list[str] = Field(
        default_factory=list,
        description="Additional output keywords, e.g. ['mulliken', 'hirshfeld', 'band_structure']",
    )
    extra_keywords: dict[str, str] = Field(
        default_factory=dict,
        description="Additional control.in keywords as key-value pairs",
    )

    @field_validator("k_grid")
    @classmethod
    def validate_k_grid(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None and len(v) != 3:
            raise ValueError(f"k_grid must have 3 components, got {len(v)}")
        return v


# ---------------------------------------------------------------------------
# Psi4 parameters
# ---------------------------------------------------------------------------

class Psi4Params(BaseModel):
    """Parameters for a Psi4 calculation."""
    method: Psi4Method = Field(default=Psi4Method.HF, description="Electronic structure method")
    functional: Optional[str] = Field(
        default=None,
        description="DFT functional name (used when method=dft)",
    )
    basis: str = Field(default="cc-pVDZ", description="Basis set name")
    calculation_type: CalculationType = Field(default=CalculationType.SINGLE_POINT)
    reference: Optional[str] = Field(
        default=None,
        description="Reference wavefunction: rhf, uhf, rohf. Auto-set if None.",
    )
    frozen_core: bool = Field(
        default=True,
        description="Freeze core orbitals in correlated methods",
    )
    symmetry: Optional[str] = Field(
        default=None,
        description="Point group symmetry. None = auto-detect, 'c1' = no symmetry.",
    )
    scf_type: str = Field(default="df", description="SCF algorithm: df, pk, out_of_core, direct")
    e_convergence: float = Field(default=1e-8, description="Energy convergence (Hartree)")
    d_convergence: float = Field(default=1e-8, description="Density convergence")
    maxiter: int = Field(default=200, description="Max SCF iterations")
    memory: str = Field(default="4 GB", description="Memory allocation")
    num_threads: int = Field(default=4, description="Number of OMP threads")
    dispersion: DispersionMethod = Field(default=DispersionMethod.NONE)
    dertype: Optional[str] = Field(
        default=None,
        description="Derivative type for optimization: 'gradient' or 'energy'",
    )
    extra_options: dict[str, str] = Field(
        default_factory=dict,
        description="Additional Psi4 options as key-value pairs",
    )

    @model_validator(mode="after")
    def validate_method_functional(self) -> Psi4Params:
        if self.method == Psi4Method.DFT and not self.functional:
            raise ValueError("functional must be specified when method='dft'")
        return self


# ---------------------------------------------------------------------------
# Parsed output models
# ---------------------------------------------------------------------------

class ParsedEnergies(BaseModel):
    """Energies extracted from a calculation output."""
    total_energy: Optional[float] = Field(default=None, description="Total energy (eV)")
    total_energy_hartree: Optional[float] = Field(default=None, description="Total energy (Ha)")
    scf_energies: list[float] = Field(
        default_factory=list,
        description="SCF energy at each iteration (eV)",
    )
    correlation_energy: Optional[float] = Field(
        default=None, description="Correlation energy (Ha)",
    )
    mp2_energy: Optional[float] = Field(default=None, description="MP2 total energy (Ha)")
    ccsd_energy: Optional[float] = Field(default=None, description="CCSD total energy (Ha)")
    ccsd_t_energy: Optional[float] = Field(default=None, description="CCSD(T) total energy (Ha)")
    dispersion_energy: Optional[float] = Field(
        default=None, description="Dispersion correction energy (Ha)",
    )
    zero_point_energy: Optional[float] = Field(default=None, description="ZPE (eV)")


class ParsedForces(BaseModel):
    """Forces extracted from a calculation output."""
    forces: list[list[float]] = Field(
        default_factory=list,
        description="Atomic forces, shape (N, 3) in eV/Angstrom",
    )
    max_force: Optional[float] = Field(default=None, description="Maximum force component")
    rms_force: Optional[float] = Field(default=None, description="RMS force")


class ParsedOrbitals(BaseModel):
    """Orbital energies and occupations."""
    eigenvalues: list[float] = Field(
        default_factory=list, description="Orbital energies (eV)",
    )
    occupations: list[float] = Field(
        default_factory=list, description="Orbital occupations",
    )
    homo_index: Optional[int] = Field(default=None, description="HOMO index (0-based)")
    homo_energy: Optional[float] = Field(default=None, description="HOMO energy (eV)")
    lumo_energy: Optional[float] = Field(default=None, description="LUMO energy (eV)")
    homo_lumo_gap: Optional[float] = Field(default=None, description="HOMO-LUMO gap (eV)")


class ParsedFrequencies(BaseModel):
    """Vibrational frequencies from a frequency calculation."""
    frequencies: list[float] = Field(
        default_factory=list, description="Vibrational frequencies (cm^-1)",
    )
    intensities: list[float] = Field(
        default_factory=list, description="IR intensities (km/mol)",
    )
    is_imaginary: list[bool] = Field(
        default_factory=list, description="Whether each mode is imaginary",
    )
    n_imaginary: int = Field(default=0, description="Number of imaginary frequencies")


class OptimizationTrajectory(BaseModel):
    """Trajectory from a geometry optimization."""
    energies: list[float] = Field(default_factory=list, description="Energy at each step (eV)")
    max_forces: list[float] = Field(
        default_factory=list, description="Max force at each step (eV/Ang)",
    )
    converged: bool = Field(default=False, description="Whether optimization converged")
    n_steps: int = Field(default=0)


class CalculationResult(BaseModel):
    """Complete parsed result from a calculation."""
    program: str = Field(description="Program that produced the output (fhi-aims, psi4, gaussian)")
    calc_type: Optional[str] = Field(default=None, description="Calculation type detected")
    success: bool = Field(default=False, description="Whether the calculation completed normally")
    energies: ParsedEnergies = Field(default_factory=ParsedEnergies)
    forces: ParsedForces = Field(default_factory=ParsedForces)
    orbitals: ParsedOrbitals = Field(default_factory=ParsedOrbitals)
    frequencies: ParsedFrequencies = Field(default_factory=ParsedFrequencies)
    optimization: OptimizationTrajectory = Field(default_factory=OptimizationTrajectory)
    n_atoms: Optional[int] = Field(default=None)
    n_electrons: Optional[int] = Field(default=None)
    n_basis_functions: Optional[int] = Field(default=None)
    wall_time_seconds: Optional[float] = Field(default=None)
    warnings: list[str] = Field(default_factory=list)
    raw_metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Additional code-specific parsed data",
    )


# ---------------------------------------------------------------------------
# Generated input file models
# ---------------------------------------------------------------------------

class GeneratedInput(BaseModel):
    """Result of input file generation — contains the file contents."""
    files: dict[str, str] = Field(
        description="Mapping of filename to file content, e.g. {'control.in': '...'}",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings generated during input creation",
    )
    summary: str = Field(
        default="",
        description="Human-readable summary of the generated calculation",
    )
