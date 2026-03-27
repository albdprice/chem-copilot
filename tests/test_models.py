"""Tests for Pydantic models."""

from __future__ import annotations

import pytest

from chem_copilot.models import (
    CalculationType,
    FHIAimsParams,
    MoleculeSpec,
    Psi4Method,
    Psi4Params,
    XCFunctional,
)


class TestMoleculeSpec:
    def test_basic_construction(self, water: MoleculeSpec) -> None:
        assert water.n_atoms == 3
        assert not water.is_periodic
        assert water.charge == 0
        assert water.multiplicity == 1

    def test_from_xyz_string(self, water_xyz: MoleculeSpec) -> None:
        assert water_xyz.n_atoms == 3
        symbols, coords = water_xyz.to_symbols_coords()
        assert symbols == ["O", "H", "H"]
        assert len(coords) == 3
        assert abs(coords[0][2] - 0.1173) < 1e-4

    def test_periodic(self, nacl_crystal: MoleculeSpec) -> None:
        assert nacl_crystal.is_periodic
        assert nacl_crystal.n_atoms == 8
        assert nacl_crystal.lattice_vectors is not None
        assert len(nacl_crystal.lattice_vectors) == 3

    def test_spin(self, oxygen_triplet: MoleculeSpec) -> None:
        assert oxygen_triplet.multiplicity == 3
        assert oxygen_triplet.n_unpaired_electrons == 2

    def test_charged(self, water_charged: MoleculeSpec) -> None:
        assert water_charged.charge == -1
        assert water_charged.n_atoms == 2

    def test_mismatched_symbols_coords_raises(self) -> None:
        with pytest.raises(ValueError, match="symbols.*coords.*same length"):
            MoleculeSpec(
                symbols=["O", "H"],
                coords=[[0, 0, 0]],
            )

    def test_bad_coords_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="3 components"):
            MoleculeSpec(
                symbols=["O"],
                coords=[[0, 0]],
            )

    def test_bad_lattice_raises(self) -> None:
        with pytest.raises(ValueError, match="3 lattice vectors"):
            MoleculeSpec(
                symbols=["Na"],
                coords=[[0, 0, 0]],
                lattice_vectors=[[1, 0, 0], [0, 1, 0]],
            )

    def test_to_symbols_coords_from_fields(self, water: MoleculeSpec) -> None:
        symbols, coords = water.to_symbols_coords()
        assert symbols == ["O", "H", "H"]
        assert len(coords) == 3

    def test_xyz_takes_precedence(self) -> None:
        mol = MoleculeSpec(
            symbols=["Ne"],
            coords=[[0, 0, 0]],
            xyz_string="1\nneon\nNe 1.0 2.0 3.0\n",
        )
        symbols, coords = mol.to_symbols_coords()
        assert symbols == ["Ne"]
        assert abs(coords[0][0] - 1.0) < 1e-6


class TestFHIAimsParams:
    def test_defaults(self) -> None:
        p = FHIAimsParams()
        assert p.xc == XCFunctional.PBE
        assert p.calculation_type == CalculationType.SINGLE_POINT

    def test_k_grid_validation(self) -> None:
        with pytest.raises(ValueError, match="3 components"):
            FHIAimsParams(k_grid=[4, 4])


class TestPsi4Params:
    def test_defaults(self) -> None:
        p = Psi4Params()
        assert p.method == Psi4Method.HF
        assert p.basis == "cc-pVDZ"

    def test_dft_requires_functional(self) -> None:
        with pytest.raises(ValueError, match="functional must be specified"):
            Psi4Params(method=Psi4Method.DFT)

    def test_dft_with_functional(self) -> None:
        p = Psi4Params(method=Psi4Method.DFT, functional="b3lyp")
        assert p.functional == "b3lyp"
