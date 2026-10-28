"""Parsers on the tiny synthetic aims outputs in tests/synthetic/."""
from pathlib import Path

from aimshalf.aims_io import AimsStdout, BandStructure, atoms_map

S = Path(__file__).parent / "synthetic"


def test_band_edges_from_band_files():
    e = BandStructure.from_dir(str(S)).band_edges()
    assert abs(e["vbm"] + 2.1) < 1e-9 and abs(e["cbm"] - 1.3) < 1e-9
    assert abs(e["gap"] - 3.4) < 1e-9 and e["direct"]


def test_gap_from_aims_out():
    assert abs(AimsStdout(str(S / "aims.out")).gap_ev - 3.4) < 1e-9


def test_geometry_species():
    assert atoms_map(str(S / "geometry.in")) == {1: "Ga", 2: "Ga", 3: "N", 4: "N"}
