"""divide_character, as minushalf: an atom with n nearest neighbours of its own species
shares the hole and gets 1/(1+n) of its share (Si: 1/4 e)."""
import shutil
from pathlib import Path

from aimshalf.aims_io import equal_neighbors
from aimshalf.workflow import select_channels

T = Path(__file__).parent


def test_equal_neighbors():
    assert equal_neighbors(str(T / "synthetic_si" / "geometry.in"), "Si") == 1
    assert equal_neighbors(str(T / "synthetic_mlk" / "geometry.in"), "As") == 0
    assert equal_neighbors(str(T / "synthetic_mlk" / "geometry.in"), "Al") == 0


def test_divide_character_in_selection(tmp_path):
    ref = tmp_path / "pbe_ref"
    shutil.copytree(T / "synthetic_mlk", ref)
    ch, _ = select_channels(ref, "v")
    assert ch[0]["dq"] == 0.5                                  # III-V: n = 0
    ch, txt = select_channels(ref, "v", divide_character=[["As", "p", 1]])
    assert ch[0]["dq"] == 0.25 and "1/2 of its share" in txt   # forced n = 1
