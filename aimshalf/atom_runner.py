"""
Driver for the bundled ATOM program (atm_cGuima3, standalone build `atmhalf`).

INP format (all-electron 'ae' calculation), as in the minushalf fixtures:

       ae      Ga
     n=Ga c=pb
           0.0       0.0 ...
        5    4                     <- n_core_orbitals  n_valence_orbitals
        4    0     2.000     0.000 <- n  l  occ_down  occ_up  (2i5,2f10.3)
        ...
    100 maxit

XC codes ('c=' field): 'ca' Ceperley-Alder LDA, 'pb' PBE, 'wi' Wigner,
'hl' Hedin-Lundqvist, 'gl' Gunnarson-Lundqvist, 'bh' von Barth-Hedin,
'rp' RPBE, 'rv' revPBE, 'bl' BLYP. Append 'r' (e.g. 'pbr') for the
relativistic (Dirac) solution.

IMPORTANT consistency note: match the XC code to the functional used in
FHI-aims (pb <-> pbe, ca <-> pw-lda approx.). For heavy elements FHI-aims
uses atomic ZORA by default while ATOM's relativistic mode solves the
Dirac equation; the SEP is a potential *difference*, which cancels much of
the discrepancy, but validate against a WIEN2k/exciting reference for
heavy species.
"""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .sep import Vtotal

# atom/ (solver + inputs) lives in the source folder: install with `pip install -e .`,
# or point AIMSHALF_ROOT to the source folder.
PKG_ROOT = Path(os.environ.get("AIMSHALF_ROOT", Path(__file__).resolve().parent.parent))
DEFAULT_EXE = PKG_ROOT / "atom" / "atmhalf"
FIXTURES = PKG_ROOT / "atom" / "fixtures"

_ORBITAL_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")
L_LABEL = {"s": 0, "p": 1, "d": 2, "f": 3}


class AtomInp:
    """Minimal INP reader/writer supporting occupation edits."""

    def __init__(self, lines):
        self.lines = [ln.rstrip("\n") for ln in lines]
        # locate the orbital block: line with two ints (ncore nval) followed
        # by nval orbital lines
        self.header_idx = None
        for i, ln in enumerate(self.lines):
            m = re.match(r"^\s*(\d+)\s+(\d+)\s*$", ln)
            if m:
                self.header_idx = i
                self.ncore, self.nval = int(m.group(1)), int(m.group(2))
                break
        if self.header_idx is None:
            raise ValueError("Could not find orbital-count line in INP")
        self.orbitals = []  # list of dicts
        for j in range(self.header_idx + 1, self.header_idx + 1 + self.nval):
            m = _ORBITAL_RE.match(self.lines[j])
            if not m:
                raise ValueError(f"Bad orbital line in INP: {self.lines[j]!r}")
            self.orbitals.append({
                "line": j, "n": int(m.group(1)), "l": int(m.group(2)),
                "occ_down": float(m.group(3)), "occ_up": float(m.group(4)),
            })

    @classmethod
    def for_species(cls, symbol):
        path = FIXTURES / symbol / "INP"
        if not path.exists():
            raise FileNotFoundError(
                f"No INP fixture for species {symbol} at {path}. aimshalf reads its "
                f"atomic inputs from the source folder: install it with 'pip install -e .' "
                f"there, or set AIMSHALF_ROOT to that folder.")
        return cls(path.read_text().splitlines())

    def set_xc(self, code):
        for i, ln in enumerate(self.lines):
            if "c=" in ln:
                self.lines[i] = re.sub(r"c=\S+", f"c={code}", ln)
                return
        raise ValueError("No 'c=' XC field found in INP")

    def remove_charge(self, l, dq):
        """Remove dq electrons from the valence channel with angular momentum l."""
        for orb in self.orbitals:
            if orb["l"] == l:
                if orb["occ_down"] < dq - 1e-9:
                    raise ValueError(
                        f"Cannot remove {dq} e from l={l}: occ={orb['occ_down']}")
                orb["occ_down"] -= dq
                self._rewrite(orb)
                return
        raise ValueError(f"No valence orbital with l={l} in INP")

    def _rewrite(self, orb):
        self.lines[orb["line"]] = "{:5d}{:5d}{:10.3f}{:10.3f}".format(
            orb["n"], orb["l"], orb["occ_down"], orb["occ_up"])

    def text(self):
        return "\n".join(self.lines) + "\n"


def run_atom(inp_text, exe=DEFAULT_EXE, workdir=None) -> Vtotal:
    """Run the ATOM program on the given INP text; return parsed VTOTAL."""
    exe = Path(exe)
    if not exe.exists():
        raise FileNotFoundError(
            f"atmhalf executable not found at {exe}; build it with "
            f"`make atom` (see README)")
    tmp = workdir or tempfile.mkdtemp(prefix="aimshalf_atom_")
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "INP").write_text(inp_text)
    res = subprocess.run([str(exe)], cwd=tmp, capture_output=True, text=True,
                         timeout=300)
    vt = tmp / "VTOTAL1"
    if not vt.exists():
        raise RuntimeError(
            f"ATOM run produced no VTOTAL1 in {tmp}\nstdout:\n{res.stdout[-2000:]}")
    vtotal = Vtotal.from_file(str(vt))
    if workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return vtotal


def make_sep_inputs(symbol, corrections, xc="pb"):
    """
    corrections: dict {l (int or 's'/'p'/'d'/'f'): charge_removed}
    e.g. {'p': 0.5} for a plain anion -1/2, or {'s': 0.111, 'p': 0.389}
    for a projection-weighted split (fractional correction).
    Returns (inp_neutral_text, inp_ion_text).
    """
    neutral = AtomInp.for_species(symbol)
    neutral.set_xc(xc)
    ion = AtomInp.for_species(symbol)
    ion.set_xc(xc)
    for l, dq in corrections.items():
        l_int = L_LABEL[l] if isinstance(l, str) else int(l)
        ion.remove_charge(l_int, dq)
    return neutral.text(), ion.text()
