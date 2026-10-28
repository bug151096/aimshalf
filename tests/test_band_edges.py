"""Band edges as minushalf defines them: smallest direct gap (default) or fundamental
gap (indirect), and the character at a chosen (k-point, state).

Synthetic AlAs-like case (tests/synthetic_mlk): VBM at k1 (-2.0 eV); CBM at k2 (1.0 eV).
Fundamental gap k1 -> k2 = 3.0 eV; direct gaps 3.5 (k1) and 4.0 (k2) eV.
"""
from pathlib import Path

from aimshalf.aims_io import BandStructure, edge_character, character_table, species_character, atoms_map
from aimshalf.cli import main

S = Path(__file__).parent / "synthetic_mlk"
MLK = [str(S / "bandmlk1001.out")]


def species(info_proj):
    _, proj = info_proj
    return species_character(character_table(proj, atoms_map(str(S / "geometry.in"))))


def test_gap_fundamental_and_direct():
    bs = BandStructure.from_dir(str(S))
    ind = bs.band_edges(indirect=True)
    assert abs(ind["gap"] - 3.0) < 1e-9 and not ind["direct"]
    assert (ind["vbm_k"], ind["cbm_k"]) == (0, 1)
    dirg = bs.band_edges(indirect=False)
    assert abs(dirg["gap"] - 3.5) < 1e-9 and dirg["direct"] and dirg["cbm_k"] == 0


def test_edge_character_modes():
    info, _ = edge_character(MLK, "cbm", indirect=True)
    assert info["k_number"] == 2                       # fundamental: CBM at k2
    sc = species(edge_character(MLK, "cbm", indirect=False))
    assert abs(sc[("Al", "s")] - 60.0) < 1e-6         # direct: CBM at k1, s-like
    sc = species(edge_character(MLK, "vbm", indirect=False))
    assert next(iter(sc)) == ("As", "p")


def test_band_character_location():
    info, _ = edge_character(MLK, location=(2, 2))
    assert info["k_number"] == 2 and info["states"] == [2] and info["energy"] == 1.0
    sc = species(edge_character(MLK, location=(2, 2)))
    assert abs(sc[("Al", "p")] - 50.0) < 1e-6


def test_cli_band_commands(capsys):
    main(["band-gap", "--dir", str(S)])
    assert "3.5000 eV (direct)" in capsys.readouterr().out
    main(["band-gap", "--dir", str(S), "-i"])
    assert "3.0000 eV (indirect)" in capsys.readouterr().out
    main(["band-character", "2", "2", "--dir", str(S)])
    assert "k-point 2" in capsys.readouterr().out
