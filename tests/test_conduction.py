"""Conduction corrections, as minushalf: codes c, cf, vc, vfc, vcf, vfcf; conduction
channels come after the valence ones and enter the SEP with amplitude -|A|.

Synthetic case (tests/synthetic_mlk), smallest direct gap at k1:
VBM As-p 70 / Al-p 20 / Al-d 10 %;  CBM Al-s 60 / As-s 40 %.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from aimshalf.atom_runner import DEFAULT_EXE
from aimshalf.workflow import select_channels, parse_correction_code, write_species_sep

T = Path(__file__).parent


def pick(tmp_path, code, **kw):
    ref = tmp_path / "pbe_ref"
    if not ref.exists():
        shutil.copytree(T / "synthetic_mlk", ref)
    ch, _ = select_channels(ref, code, indirect=False, **kw)
    return [(c["species"], "spdf"[c["l"]], c["dq"], c["conduction"]) for c in ch]


def test_codes(tmp_path):
    assert pick(tmp_path, "v") == [("As", "p", 0.5, False)]
    assert pick(tmp_path, "c") == [("Al", "s", 0.5, True)]
    assert pick(tmp_path, "vc") == [("As", "p", 0.5, False), ("Al", "s", 0.5, True)]
    assert pick(tmp_path, "vfcf") == [("As", "p", 0.39, False), ("Al", "p", 0.11, False),
                                      ("Al", "s", 0.30, True), ("As", "s", 0.20, True)]
    # cbm_characters replaces CBM values before the selection
    assert pick(tmp_path, "c", cbm_characters=[["As", "s", 90]]) == [("As", "s", 0.5, True)]


def test_invalid_code():
    with pytest.raises(ValueError):
        parse_correction_code("vfx")


@pytest.mark.skipif(not DEFAULT_EXE.exists(), reason="atmhalf not built (run: make atom)")
def test_conduction_sep_is_minus_valence(tmp_path):
    def read(p):
        lines = [l for l in open(p) if not l.startswith("#")]
        return np.array([l.split() for l in lines[1:]], float)
    v = write_species_sep(tmp_path, "As", [{"species": "As", "l": 0, "dq": 0.5, "cut": 3.0}])
    (tmp_path / "c").mkdir()
    c = write_species_sep(tmp_path / "c", "As",
                          [{"species": "As", "l": 0, "dq": 0.5, "cut": 3.0, "conduction": True}])
    assert Path(c).name == "sep_As_ccut3.000.dat"
    a, b = read(v), read(c)
    assert np.allclose(a[:, 0], b[:, 0]) and np.allclose(b[:, 1], -a[:, 1])
    assert b[:, 1].max() > 0                     # repulsive
