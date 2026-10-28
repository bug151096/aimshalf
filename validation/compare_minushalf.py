#!/usr/bin/env python3
"""
compare_minushalf.py -- validate aimshalf's self-energy potential (SEP)
against the minushalf reference implementation (ITA group).

Drop this file in  aims_half/validation/  and run it with no arguments:

    python3 compare_minushalf.py

It finds both repositories by itself (aims_half = parent of this folder,
minushalf = sibling folder "minushalf"; override with the environment
variables AIMSHALF_ROOT / MINUSHALF_ROOT if your layout differs), imports
minushalf's OWN Python source (nothing is re-implemented on the minushalf
side), and writes everything to  validation/results/ :

    report.txt                 pass/fail table + numbers
    sep_overlay_<case>.png     aimshalf vs minushalf SEP, plus |difference|
    fixture_vs_fresh_Ga.png    minushalf-shipped VTOTALs vs a fresh atmhalf run
    fourier_check.png          minushalf's q-space POTCAR term vs FT of aimshalf's SEP
    split_vs_separate.png      multi-channel SEP: aimshalf file vs minushalf per-channel sum

Checks (Layer 1 = the decisive, engine-independent ones):
  L1-a  INP generation: minushalf InputFile(minimum_setup + electron_occupation)
        vs aimshalf AtomInp(fixture + remove_charge), every element, every l.
  L1-b  Atomic solver: the two INPs run through atmhalf give identical VTOTALs;
        minushalf's shipped Ga VTOTAL/VTOTAL_OCC are reproduced by atmhalf.
  L1-c  VTOTAL parsing: minushalf Vtotal vs aimshalf Vtotal on the same file.
  L1-d  SEP: minushalf AtomicPotential.occupy_potential (trimming_function)
        vs aimshalf SelfEnergyPotential.trimmed, after unit conversion,
        several cuts, valence (A=+1) and conduction (A=-1).
  L1-e  The q-space correction minushalf adds to the POTCAR equals the radial
        Fourier transform of the real-space SEP aimshalf hands to FHI-aims
        (i.e. both inject the same physical potential).
  L2    Multi-channel corrections (e.g. As s:0.1 + p:0.4): the SEP file aimshalf
        writes (write_species_sep, used by execute and gen-sep) vs the sum of
        minushalf's per-channel SEPs (one atomic run per channel), same r_cut.
"""
import os
import sys
import types
import shutil
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# settings (edit here if needed; no command-line arguments by design)
# ----------------------------------------------------------------------------
SEP_CASES = [            # (species, l, electrons removed)
    ("As", 1, 0.5),      # the GaAs production case (anion p, valence)
    ("Ga", 1, 0.5),      # same as minushalf's shipped Ga fixture
    ("Ga", 0, 0.5),      # cation s (conduction-type channel)
    ("N", 1, 0.5),
    ("P", 1, 0.5),
    ("In", 1, 0.5),
]
CUTS = [2.0, 3.0, 3.8, 5.0]           # bohr
AMPLITUDES = [1.0, -1.0]              # valence, conduction
XC = "pb"
SPLIT_CASE = ("As", {0: 0.1, 1: 0.4}) # 20% s / 80% p of half an electron
PLOT_CUT = 3.8

TOL_SEP_SAME_INPUT = 1e-10   # Ha  -- same VTOTALs into both builders: must be ~float noise
TOL_ATOM_REPRO = 1e-5        # Ha  -- fixture (other build) vs fresh atmhalf run
TOL_FT_REL = 1e-3            # relative -- FT check (quadrature differences only)

HA_EV = 27.211386245988
BOHR_A = 0.529177210903

# ----------------------------------------------------------------------------
# locate repositories
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent


def _find_root(env, candidates, marker):
    if os.environ.get(env):
        p = Path(os.environ[env]).expanduser().resolve()
        if (p / marker).exists():
            return p
        sys.exit(f"{env}={p} does not contain {marker}")
    for c in candidates:
        if (c / marker).exists():
            return c.resolve()
    sys.exit(f"Could not find {marker}. Set {env} to the repository root.")


AIMS_ROOT = _find_root("AIMSHALF_ROOT",
                       [HERE.parent, HERE, HERE.parent.parent / "aims_half"],
                       "aimshalf/sep.py")
MH_ROOT = _find_root("MINUSHALF_ROOT",
                     [AIMS_ROOT.parent / "minushalf", HERE.parent.parent / "minushalf",
                      HERE.parent / "minushalf"],
                     "minushalf/utils/trimming_function.py")
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# minimal stand-ins for minushalf's non-numeric dependencies, used ONLY if the
# real packages are not installed (logging + Fortran-style line formatting)
# ----------------------------------------------------------------------------
try:
    import loguru  # noqa: F401
except ImportError:
    _lg = types.ModuleType("loguru")

    class _L:
        def __getattr__(self, _):
            return lambda *a, **k: None
    _lg.logger = _L()
    sys.modules["loguru"] = _lg

try:
    import fortranformat  # noqa: F401
except ImportError:
    import re as _re
    _ff = types.ModuleType("fortranformat")

    class FortranRecordWriter:
        """Subset of Fortran edit descriptors used by minushalf: nX, rIw, rFw.d, rAw."""

        def __init__(self, fmt):
            self.items = []
            for tok in fmt.replace(" ", "").split(","):
                m = _re.fullmatch(r"(\d*)x", tok, _re.I)
                if m:
                    self.items.append(("x", int(m.group(1) or 1), None))
                    continue
                m = _re.fullmatch(r"(\d*)([ifa])(\d+)(?:\.(\d+))?", tok, _re.I)
                if not m:
                    raise ValueError(f"unsupported descriptor {tok}")
                rep = int(m.group(1) or 1)
                for _ in range(rep):
                    self.items.append((m.group(2).lower(), int(m.group(3)),
                                       int(m.group(4)) if m.group(4) else None))

        def write(self, values):
            vals = list(values)
            out = []
            for kind, w, d in self.items:
                if kind == "x":
                    out.append(" " * w)
                    continue
                if not vals:
                    break
                v = vals.pop(0)
                if kind == "i":
                    s = f"{int(v):>{w}d}"
                elif kind == "f":
                    s = f"{float(v):>{w}.{d}f}"
                else:
                    s = f"{str(v):>{w}}"[:w]
                out.append("*" * w if len(s) > w else s)
            return "".join(out)
    _ff.FortranRecordWriter = FortranRecordWriter
    sys.modules["fortranformat"] = _ff

# ----------------------------------------------------------------------------
# import both codes from source
# ----------------------------------------------------------------------------
sys.path.insert(0, str(MH_ROOT))
sys.path.insert(0, str(AIMS_ROOT))

from minushalf.utils.trimming_function import trimming_function as mh_trimming  # noqa: E402
from minushalf.utils.correct_potential_fourier_transform import \
    correct_potential_fourier_transform as mh_ft  # noqa: E402
from minushalf.utils.constants import Constants as MhConstants  # noqa: E402
from minushalf.io.vtotal import Vtotal as MhVtotal  # noqa: E402
from minushalf.io.input_file import InputFile as MhInputFile  # noqa: E402

from aimshalf.sep import Vtotal as AhVtotal, SelfEnergyPotential  # noqa: E402
from aimshalf.atom_runner import AtomInp, DEFAULT_EXE, FIXTURES as AH_FIX  # noqa: E402

MHC = MhConstants()
MH_PREF = 4 * MHC.pi_constant * MHC.rydberg * MHC.bohr_radius ** 3
MH_FIX = MH_ROOT / "minushalf" / "fixtures"
L_NAME = "spdf"

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
REPORT = []
RESULTS = []   # (check, case, ok, detail)


def log(s=""):
    print(s)
    REPORT.append(s)


def record(check, case, ok, detail):
    RESULTS.append((check, case, ok, detail))
    log(f"  [{'PASS' if ok else 'FAIL'}] {check:5s} {case:28s} {detail}")


def run_atmhalf(inp_text, keep_dir=None):
    d = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix="cmpmh_"))
    d.mkdir(parents=True, exist_ok=True)
    (d / "INP").write_text(inp_text)
    import subprocess
    subprocess.run([str(DEFAULT_EXE)], cwd=d, capture_output=True, text=True, timeout=300)
    vt = d / "VTOTAL1"
    if not vt.exists():
        raise RuntimeError(f"atmhalf produced no VTOTAL1 in {d}")
    return vt, d


def orbital_table_mh(inp):
    return [(o["n"], o["l"], round(float(o["occupation"][0]), 6),
             round(float(o["occupation"][1]), 6)) for o in inp.valence_orbitals]


def orbital_table_ah(inp):
    return [(o["n"], o["l"], round(o["occ_down"], 6), round(o["occ_up"], 6))
            for o in inp.orbitals]


def mh_inp(symbol, removals):
    """INP exactly as minushalf's correction loop builds it."""
    inp = MhInputFile.minimum_setup(symbol, XC, 100, "ae")
    for l, dq in removals.items():
        inp.electron_occupation(dq, l)
    return inp


def ah_inp(symbol, removals):
    """INP exactly as aimshalf.make_sep_inputs builds it."""
    inp = AtomInp.for_species(symbol)
    inp.set_xc(XC)
    for l, dq in removals.items():
        inp.remove_charge(l, dq)
    return inp


def mh_sep_hartree(vt_atom_file, vt_ion_file, cut, amp):
    """minushalf's real-space SEP (occupy_potential) converted to V(r) in Ha."""
    a, i = MhVtotal.from_file(str(vt_atom_file)), MhVtotal.from_file(str(vt_ion_file))
    r = np.asarray(a.radius, float)
    if amp < 0:      # minushalf's conduction convention: amplitude -> -|A|
        amp = -abs(amp)
    p = mh_trimming(r, np.asarray(i.down_potential, float),
                    np.asarray(a.down_potential, float), cut, amp)
    return r, p, (p / MH_PREF) * 0.5 / r   # [4pi Ry a0^3 * rV(Ry bohr)] -> V(Ha)


def ah_sep_hartree(vt_atom_file, vt_ion_file, cut, amp):
    sep = SelfEnergyPotential(AhVtotal.from_file(str(vt_atom_file)),
                              AhVtotal.from_file(str(vt_ion_file)))
    return sep.radius, sep.trimmed(cut, amplitude=amp)


# ============================================================================
log("aimshalf vs minushalf -- self-energy potential validation")
log(f"aimshalf : {AIMS_ROOT}")
log(f"minushalf: {MH_ROOT}  (version {__import__('minushalf').__version__})")
log(f"atmhalf  : {DEFAULT_EXE}")
log(f"XC = {XC}, trimming exponent: minushalf {MHC.trimming_exponent}, aimshalf 8")
log("")

# ---------------------------------------------------------------- L1-a ------
log("L1-a  INP generation (all elements with fixtures in both codes, l = s,p,d,f, dq = 0.5)")
species_all = sorted(p.name for p in AH_FIX.iterdir()
                     if (p / "INP").exists() and len(p.name) <= 2 and p.name[0].isupper())
n_ok, mism = 0, []
for sym in species_all:
    try:
        base_mh = orbital_table_mh(MhInputFile.minimum_setup(sym, XC, 100, "ae"))
    except Exception as e:  # element not in minushalf's table
        mism.append(f"{sym}: minushalf cannot build INP ({e})")
        continue
    base_ah = orbital_table_ah(ah_inp(sym, {}))
    if base_mh != base_ah:
        mism.append(f"{sym} neutral: mh {base_mh} vs ah {base_ah}")
        continue
    for l in range(4):
        try:
            t_mh = orbital_table_mh(mh_inp(sym, {l: 0.5}))
        except Exception:
            t_mh = "error"
        try:
            t_ah = orbital_table_ah(ah_inp(sym, {l: 0.5}))
        except Exception:
            t_ah = "error"
        if t_mh == t_ah:
            n_ok += 1
        else:
            mism.append(f"{sym} -0.5e l={L_NAME[l]}: minushalf -> {t_mh}\n"
                        f"{'':22s}aimshalf  -> {t_ah}")
record("L1-a", f"{len(species_all)} elements x 4 l", not mism,
       f"{n_ok} identical; {len(mism)} differ")
for m in mism:
    log("        " + m)
log("")

# ---------------------------------------------------------------- L1-b/c/d --
log("L1-b/c/d  atomic runs, parsing and SEP for each case")
work = Path(tempfile.mkdtemp(prefix="cmpmh_runs_"))
case_data = {}
for sym, l, dq in SEP_CASES:
    tag = f"{sym}_{L_NAME[l]}{dq:g}"
    try:
        im_n, im_i = mh_inp(sym, {}), mh_inp(sym, {l: dq})
        ia_n, ia_i = ah_inp(sym, {}), ah_inp(sym, {l: dq})
    except Exception as e:
        record("L1-b", tag, False, f"could not build INPs: {e}")
        continue
    vt = {}
    for key, text in [("mh_n", "".join(im_n.to_stringlist())),
                      ("mh_i", "".join(im_i.to_stringlist())),
                      ("ah_n", ia_n.text()), ("ah_i", ia_i.text())]:
        vt[key], _ = run_atmhalf(text, work / tag / key)

    # L1-b: same physics from both INP writers?
    dmax = 0.0
    for a, b in [("mh_n", "ah_n"), ("mh_i", "ah_i")]:
        ra, rb = AhVtotal.from_file(str(vt[a])), AhVtotal.from_file(str(vt[b]))
        dmax = max(dmax, float(np.max(np.abs(ra.potential_hartree() - rb.potential_hartree()))))
    record("L1-b", tag + " (INP->VTOTAL)", dmax < TOL_SEP_SAME_INPUT,
           f"max|dV| between VTOTALs from mh/ah INPs = {dmax:.2e} Ha")

    # L1-c: parsers
    pm, pa = MhVtotal.from_file(str(vt["ah_i"])), AhVtotal.from_file(str(vt["ah_i"]))
    same_parse = (len(pm.radius) == len(pa.radius)
                  and np.array_equal(np.asarray(pm.radius, float), pa.radius)
                  and np.array_equal(np.asarray(pm.down_potential, float), pa.rv_down))
    record("L1-c", tag, same_parse,
           f"n_r mh={len(pm.radius)} ah={len(pa.radius)}, arrays identical={same_parse}")

    # L1-d: SEP builders on identical VTOTALs
    worst = 0.0
    for cut in CUTS:
        for amp in AMPLITUDES:
            r, _, vm = mh_sep_hartree(vt["ah_n"], vt["ah_i"], cut, amp)
            ra, va = ah_sep_hartree(vt["ah_n"], vt["ah_i"], cut, amp)
            worst = max(worst, float(np.max(np.abs(vm - va))))
    record("L1-d", tag, worst < TOL_SEP_SAME_INPUT,
           f"max|V_mh - V_ah| over cuts {CUTS}, A=+-1: {worst:.2e} Ha")
    case_data[tag] = vt

    # overlay plot
    r, _, vm = mh_sep_hartree(vt["mh_n"], vt["mh_i"], PLOT_CUT, 1.0)
    ra, va = ah_sep_hartree(vt["ah_n"], vt["ah_i"], PLOT_CUT, 1.0)
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.4]})
    ax[0].plot(ra, va * HA_EV, lw=2.5, color="#1f77b4", label="aimshalf (its own INP)")
    ax[0].plot(r, vm * HA_EV, "--", lw=1.4, color="#d62728", label="minushalf (its own INP)")
    ax[0].set_ylabel(r"$V_S(r)$ [eV]")
    ax[0].set_title(f"{sym}: -{dq:g} e from {L_NAME[l]}, r_cut = {PLOT_CUT} bohr, XC={XC}")
    ax[0].legend()
    ax[0].axvline(PLOT_CUT, color="gray", lw=0.6, ls=":")
    diff = np.abs(np.interp(ra, r, vm) - va) * HA_EV
    ax[1].semilogy(ra, np.maximum(diff, 1e-18), color="k", lw=1)
    ax[1].set_ylabel("|diff| [eV]")
    ax[1].set_xlabel("r [bohr]")
    ax[1].set_xlim(0, PLOT_CUT * 1.15)
    fig.tight_layout()
    fig.savefig(OUT / f"sep_overlay_{tag}.png", dpi=130)
    plt.close(fig)
log("")

# ------------------------------------------- L1-b: shipped minushalf fixture --
log("L1-b  minushalf's shipped Ga VTOTAL / VTOTAL_OCC (their build) vs fresh atmhalf run")
fx = MH_FIX / "Ga"
inp_occ = (fx / "INP_OCC").read_text()
vt_fresh_n, _ = run_atmhalf((fx / "INP").read_text(), work / "fixture_Ga" / "n")
vt_fresh_i, _ = run_atmhalf(inp_occ, work / "fixture_Ga" / "i")
fig, ax = plt.subplots(figsize=(6.5, 4))
worst = 0.0
for cut in CUTS:
    r1, _, v_fix = mh_sep_hartree(fx / "VTOTAL", fx / "VTOTAL_OCC", cut, 1.0)
    r2, v_new = ah_sep_hartree(vt_fresh_n, vt_fresh_i, cut, 1.0)
    if len(r1) != len(r2) or not np.allclose(r1, r2, rtol=1e-9):
        v_new = np.interp(r1, r2, v_new)
    d = np.abs(v_fix - v_new)
    worst = max(worst, float(d.max()))
    ax.semilogy(r1, np.maximum(d * HA_EV, 1e-18), label=f"r_cut={cut}")
ax.set_xlabel("r [bohr]")
ax.set_ylabel("|V_S(fixture) - V_S(fresh)| [eV]")
ax.set_title("Ga -0.5 e (4p): minushalf fixture vs fresh atmhalf + aimshalf SEP")
ax.set_xlim(0, max(CUTS) * 1.1)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fixture_vs_fresh_Ga.png", dpi=130)
plt.close(fig)
record("L1-b", "Ga fixture reproduction", worst < TOL_ATOM_REPRO,
       f"max|dV_S| = {worst:.2e} Ha = {worst*HA_EV:.2e} eV (different compiler/build)")
log("")

# ---------------------------------------------------------------- L1-e -----
log("L1-e  minushalf POTCAR q-space correction vs radial FT of aimshalf's real-space SEP")
tag0 = next(iter(case_data))
vt = case_data[tag0]
k = np.linspace(0.0, 25.0, 1000)            # 1/Angstrom, typical POTCAR range
fig, ax = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1.4]})
worst_rel = 0.0
for cut in [3.0, PLOT_CUT]:
    r, p, _ = mh_sep_hartree(vt["ah_n"], vt["ah_i"], cut, 1.0)
    term_mh = mh_ft(np.zeros_like(k), k.copy(), r, p, cut)       # added to POTCAR (eV A^3)
    ra, va = ah_sep_hartree(vt["ah_n"], vt["ah_i"], cut, 1.0)    # what aims gets (Ha, bohr)
    rA, vE = ra * BOHR_A, va * HA_EV
    m = ra <= cut
    q = k.copy()
    q[0] = 1e-12
    integrand = (rA[m] * vE[m])[None, :] * np.sin(q[:, None] * rA[m][None, :])
    ft_ah = 4 * np.pi / q * np.trapezoid(integrand, rA[m], axis=1) \
        if hasattr(np, "trapezoid") else 4 * np.pi / q * np.trapz(integrand, rA[m], axis=1)
    scale = np.max(np.abs(ft_ah))
    rel = np.max(np.abs(term_mh - ft_ah)) / scale
    worst_rel = max(worst_rel, rel)
    ax[0].plot(k, ft_ah, lw=2.5, label=f"FT of aimshalf SEP, r_cut={cut}")
    ax[0].plot(k, term_mh, "--", lw=1.3, label=f"minushalf POTCAR term, r_cut={cut}")
    ax[1].semilogy(k, np.maximum(np.abs(term_mh - ft_ah) / scale, 1e-18), lw=1)
ax[0].set_ylabel(r"$\Delta V(q)$ [eV $\AA^3$]")
ax[0].set_title(f"{tag0}: same physical potential in real and reciprocal space?")
ax[0].legend(fontsize=8)
ax[1].set_ylabel("rel. diff")
ax[1].set_xlabel(r"q [$\AA^{-1}$]")
fig.tight_layout()
fig.savefig(OUT / "fourier_check.png", dpi=130)
plt.close(fig)
record("L1-e", tag0, worst_rel < TOL_FT_REL,
       f"max relative difference = {worst_rel:.2e} (quadrature only)")
log("")

# ---------------------------------------------------------------- L2 -------
sym, split = SPLIT_CASE
log(f"L2  multi-channel correction {sym} {', '.join(f'{L_NAME[l]}:{v}' for l, v in split.items())} "
    f"at r_cut={PLOT_CUT}")
d2 = work / "split"
d2.mkdir(parents=True, exist_ok=True)
vt_n, _ = run_atmhalf(ah_inp(sym, {}).text(), d2 / "n")
# minushalf: one atomic run and one SEP per channel, summed
r, v_sum, parts = None, None, {}
for l, dq in split.items():
    vt_l, _ = run_atmhalf("".join(mh_inp(sym, {l: dq}).to_stringlist()), d2 / f"l{l}")
    r, _, vl = mh_sep_hartree(vt_n, vt_l, PLOT_CUT, 1.0)
    parts[l] = vl
    v_sum = vl.copy() if v_sum is None else v_sum + vl
# aimshalf: the SEP file that execute / gen-sep actually write for these channels
from aimshalf.workflow import write_species_sep  # noqa: E402
chans = [{"species": sym, "l": l, "dq": dq, "cut": PLOT_CUT} for l, dq in split.items()]
sep_file = write_species_sep(d2, sym, chans, xc=XC)
rows = [ln.split() for ln in open(sep_file) if not ln.startswith("#")][1:]
r_ah, v_ah = np.array(rows, float).T
m = (r_ah <= PLOT_CUT) & (r_ah >= 0.05)   # skip the rV/r numerical spike at r -> 0
dmax = float(np.max(np.abs(v_ah - np.interp(r_ah, r, v_sum))[m]))
record("L2", f"{sym} " + "+".join(L_NAME[l] for l in split), dmax < TOL_SEP_SAME_INPUT,
       f"max|V_aimshalf file - sum(V_minushalf channels)| = {dmax:.2e} Ha")
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(r_ah, v_ah * HA_EV, lw=2.5, label="aimshalf SEP file (sum of channels)")
ax.plot(r, v_sum * HA_EV, "--", lw=1.5, label="minushalf: sum of per-channel SEPs")
for l, vl in parts.items():
    ax.plot(r, vl * HA_EV, ":", lw=1, label=f"  channel {L_NAME[l]} ({split[l]} e)")
ax.set_xlim(0, PLOT_CUT * 1.1)
ax.set_xlabel("r [bohr]")
ax.set_ylabel(r"$V_S$ [eV]")
ax.set_title(f"{sym}: multi-channel correction, r_cut={PLOT_CUT} bohr")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "split_vs_separate.png", dpi=130)
plt.close(fig)
log("")

# ---------------------------------------------------------------- summary --
n_fail = sum(1 for *_, ok, _ in RESULTS if not ok)
log("=" * 78)
log(f"SUMMARY: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
for check, case, ok, detail in RESULTS:
    if not ok:
        log(f"  FAILED {check} {case}: {detail}")
log(f"results in {OUT}")
(OUT / "report.txt").write_text("\n".join(REPORT) + "\n")
shutil.rmtree(work, ignore_errors=True)
sys.exit(1 if n_fail else 0)
