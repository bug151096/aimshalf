"""Run-folder reuse: same inputs -> reused; changed or unverified -> rerun, old folder kept."""
from pathlib import Path

from aimshalf.aims_io import AimsStdout
from aimshalf.workflow import GapObjective, Progress, FINGERPRINT_FILE


class FakeRunner:
    command = ["fake_aims.x"]
    heartbeat = None

    def __init__(self):
        self.calls = 0

    def run(self, cwd, timeout=None):
        self.calls += 1
        out = Path(cwd) / "aims.out"
        out.write_text("  Overall HOMO-LUMO gap:    1.50000000 eV.\n"
                       "          Have a nice day.\n")
        return AimsStdout(str(out))


def setup(tmp_path):
    (tmp_path / "control.in").write_text("xc pbe\n")
    (tmp_path / "geometry.in").write_text("atom 0 0 0 X\n")
    amp = {"v": 1.0}

    def builder(run_dir, r_cut):
        p = Path(run_dir) / f"sep_X_cut{r_cut:.3f}.dat"
        p.write_text(f"amplitude {amp['v']}\n")
        return {"X": str(p)}
    return amp, builder


def objective(tmp_path, runner, builder, **kw):
    return GapObjective(tmp_path, None, runner=runner, sep_builder=builder,
                        progress=Progress(tmp_path, quiet=True), **kw)


def test_same_inputs_reused_changed_inputs_rerun(tmp_path):
    amp, builder = setup(tmp_path)
    runner = FakeRunner()
    assert objective(tmp_path, runner, builder)(3.0) == 1.5
    assert runner.calls == 1
    assert (tmp_path / "cut_3.000" / FINGERPRINT_FILE).exists()
    objective(tmp_path, runner, builder)(3.0)
    assert runner.calls == 1                       # same inputs: reused
    amp["v"] = 0.5
    objective(tmp_path, runner, builder)(3.0)
    assert runner.calls == 2                       # SEP changed: rerun
    assert len(list(tmp_path.glob("stale_cut_3.000_*"))) == 1
    assert not list(tmp_path.glob(".staging_*"))


def test_old_run_without_fingerprint(tmp_path):
    _, builder = setup(tmp_path)
    old = tmp_path / "cut_3.000"
    old.mkdir()
    (old / "aims.out").write_text("  Overall HOMO-LUMO gap:    9.00000000 eV.\n"
                                  "          Have a nice day.\n")
    runner = FakeRunner()
    assert objective(tmp_path, runner, builder, reuse_unverified=True)(3.0) == 9.0
    assert runner.calls == 0                       # opt-in: legacy run reused
    assert objective(tmp_path, runner, builder)(3.0) == 1.5
    assert runner.calls == 1                       # default: legacy run redone
    assert len(list(tmp_path.glob("stale_cut_3.000_*"))) == 1
