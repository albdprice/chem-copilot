"""Shared test fixtures for chem-copilot."""

from __future__ import annotations

import pytest

from chem_copilot.models import MoleculeSpec


@pytest.fixture
def water() -> MoleculeSpec:
    """Water molecule."""
    return MoleculeSpec(
        symbols=["O", "H", "H"],
        coords=[
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.7572, -0.4692],
            [0.0000, -0.7572, -0.4692],
        ],
    )


@pytest.fixture
def water_charged() -> MoleculeSpec:
    """Hydroxide anion (OH-)."""
    return MoleculeSpec(
        symbols=["O", "H"],
        coords=[
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.9700],
        ],
        charge=-1,
    )


@pytest.fixture
def methane() -> MoleculeSpec:
    """Methane molecule."""
    return MoleculeSpec(
        symbols=["C", "H", "H", "H", "H"],
        coords=[
            [0.0000, 0.0000, 0.0000],
            [0.6276, 0.6276, 0.6276],
            [-0.6276, -0.6276, 0.6276],
            [-0.6276, 0.6276, -0.6276],
            [0.6276, -0.6276, -0.6276],
        ],
    )


@pytest.fixture
def oxygen_triplet() -> MoleculeSpec:
    """O2 in triplet state."""
    return MoleculeSpec(
        symbols=["O", "O"],
        coords=[
            [0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.2075],
        ],
        multiplicity=3,
    )


@pytest.fixture
def nacl_crystal() -> MoleculeSpec:
    """NaCl rock salt unit cell."""
    a = 5.64
    return MoleculeSpec(
        symbols=["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"],
        coords=[
            [0.0, 0.0, 0.0],
            [a / 2, 0.0, 0.0],
            [a / 2, a / 2, 0.0],
            [0.0, a / 2, 0.0],
            [0.0, 0.0, a / 2],
            [a / 2, 0.0, a / 2],
            [a / 2, a / 2, a / 2],
            [0.0, a / 2, a / 2],
        ],
        lattice_vectors=[
            [a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, a],
        ],
    )


@pytest.fixture
def water_xyz() -> MoleculeSpec:
    """Water from XYZ string."""
    return MoleculeSpec(
        xyz_string="""\
3
water
O   0.0000   0.0000   0.1173
H   0.0000   0.7572  -0.4692
H   0.0000  -0.7572  -0.4692
""",
    )
