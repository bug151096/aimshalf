"""gen-sep with several -c options = sum of the single-channel SEPs (same scheme as execute)."""
import numpy as np
import pytest

from aimshalf.atom_runner import DEFAULT_EXE
from aimshalf.cli import main

pytestmark = pytest.mark.skipif(not DEFAULT_EXE.exists(),
                                reason="atmhalf not built (run: make atom)")


def read_sep(path):
    lines = [l for l in open(path) if not l.startswith("#")]
    return np.array([l.split() for l in lines[1:]], float)


def test_gen_sep_multichannel_is_sum_of_channels(tmp_path):
    main(["gen-sep", "As", "-c", "s=0.1", "--cut", "3.8", "--out-dir", str(tmp_path / "s")])
    main(["gen-sep", "As", "-c", "p=0.4", "--cut", "3.8", "--out-dir", str(tmp_path / "p")])
    main(["gen-sep", "As", "-c", "s=0.1", "-c", "p=0.4", "--cut", "3.8",
          "--out-dir", str(tmp_path / "sp")])
    s = read_sep(tmp_path / "s" / "sep_As_cut3.800.dat")
    p = read_sep(tmp_path / "p" / "sep_As_cut3.800.dat")
    sp = read_sep(tmp_path / "sp" / "sep_As_s3.800_p3.800.dat")
    assert np.allclose(sp[:, 0], s[:, 0])
    assert np.allclose(sp[:, 1], s[:, 1] + p[:, 1], atol=1e-12)
    assert sp[:, 1].min() < 0          # attractive
