"""fhiaims/install_hooks.py on a synthetic stand-in of the five FHI-aims files
(anchor lines only). The real check -- byte-identical to the former reference patch on
pristine FHI-aims 260331.1 and 261002 -- needs the FHI-aims sources and was done by hand."""
import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("install_hooks", ROOT / "fhiaims" / "install_hooks.py")
ih = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ih)


def tree(tmp_path):
    t = tmp_path / "aims"
    shutil.copytree(Path(__file__).parent / "synthetic_aims", t)
    return t


def snapshot(t):
    return {p.name: p.read_text() for p in sorted((t / "src").iterdir())}


def test_install_remove_roundtrip(tmp_path):
    t = tree(tmp_path)
    before = snapshot(t)
    ih.main([str(t)])
    src = t / "src"
    assert (src / "external_atomic_potential.f90").exists()
    cm = (src / "CMakeLists.txt").read_text().splitlines()
    i = cm.index("target_sources(aims4 PRIVATE")              # module in aims4, not aims3
    assert cm[i + 2] == "  external_atomic_potential.f90"
    rc = (src / "read_control.f90").read_text()
    assert rc.index("case ('aimshalf_potential'") < rc.index("case ('xc_pre')")
    ps = (src / "prepare_scf.f90").read_text().splitlines()
    k = ps.index("      call external_atomic_potential_initialize()")
    assert k > ps.index("          call libmbd_update_geometry()")  # second get_times only
    after = snapshot(t)
    ih.main([str(t)])                                          # idempotent
    assert snapshot(t) == after
    ih.main([str(t), "--remove"])
    assert snapshot(t) == before


def test_check_writes_nothing(tmp_path):
    t = tree(tmp_path)
    before = snapshot(t)
    ih.main([str(t), "--check"])
    assert snapshot(t) == before


def test_refuses_modified_tree(tmp_path):
    t = tree(tmp_path)
    rc = t / "src" / "read_control.f90"
    rc.write_text(rc.read_text().replace("   use warnings", "\tuse external_atomic_potential\n   use warnings"))
    before = snapshot(t)
    with pytest.raises(SystemExit):
        ih.main([str(t)])
    assert snapshot(t) == before
