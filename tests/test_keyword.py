"""control.in keyword written by aimshalf, and the check that the aims binary knows it."""
from pathlib import Path

from aimshalf.cli import _binary_contains
from aimshalf.workflow import patch_control, AIMS_KEYWORD


def test_patch_control_writes_new_keyword_and_drops_old(tmp_path):
    c = tmp_path / "control.in"
    c.write_text("xc pbe\nexternal_atomic_potential As old.dat\naimshalf_potential Ga older.dat\n")
    patch_control(c, {"As": str(tmp_path / "sep_As_cut3.800.dat")})
    text = c.read_text()
    assert AIMS_KEYWORD == "aimshalf_potential"
    assert "aimshalf_potential As sep_As_cut3.800.dat" in text
    assert "external_atomic_potential" not in text and "older.dat" not in text


def test_binary_contains(tmp_path):
    f = tmp_path / "aims.x"
    f.write_bytes(b"\x00" * (1 << 22) + b"aimshalf_potential" + b"\x00" * 10)
    assert _binary_contains(f, b"aimshalf_potential", chunk=1 << 22)
    assert _binary_contains(f, b"aimshalf_potential", chunk=7)      # across chunk borders
    assert not _binary_contains(f, b"external_atomic_potential")
