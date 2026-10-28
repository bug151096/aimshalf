"""
DFT-1/2 workflow orchestration for FHI-aims.

The correction loop mirrors minushalf's `execute`:
  1. plain PBE run -> band edges + Mulliken characters at VBM (and CBM)
  2. decide corrected species + per-orbital charge removal (simple: -1/2
     from the dominant anion channel; fractional: split by projections)
  3. for each trial r_cut: build trimmed SEP file(s), run aims with the
     aimshalf_potential keyword, read the gap
  4. maximize the gap over r_cut (coarse scan + golden-section)

Progress is visible while it runs, in the working directory:
  aimshalf_status.txt    one-glance state: RUNNING / DONE / FAILED, current cut,
                         best so far, PID   (cat it any time)
  aimshalf_execute.log   timestamped log, one block per aims run (tail -f it)
  aimshalf_results.dat   r_cut  gap  wall-time  state  -- updated after every run

aims-side requirement: the aimshalf patch (fhiaims/) applied to your
aims build. `control.in` gets one extra line per corrected species:

    aimshalf_potential <species_symbol> <sep_file>
"""
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .atom_runner import make_sep_inputs, run_atom
from .sep import SelfEnergyPotential
from .aims_io import AimsStdout, BandStructure

GOLDEN = (5 ** 0.5 - 1) / 2
AIMS_KEYWORD = "aimshalf_potential"     # control.in keyword added by the FHI-aims patch
KEYWORD_LINE = r"^\s*(aimshalf_potential|external_atomic_potential)\s.*$"   # new + old name
AIMS_DONE_MARK = "Have a nice day"


# ----------------------------------------------------------------- progress

class Progress:
    """Writes the log, the status file and the results table in `workdir`."""

    def __init__(self, workdir, quiet=False):
        self.dir = Path(workdir)
        self.log_path = self.dir / "aimshalf_execute.log"
        self.status_path = self.dir / "aimshalf_status.txt"
        self.results_path = self.dir / "aimshalf_results.dat"
        self.quiet = quiet
        self.t0 = time.time()
        self.started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.n_done = 0
        self.current = "-"
        self.best = None          # (cut, gap)
        self.rows = []            # (cut, gap, seconds, state, step)
        self.step = ""            # label of the current channel step (multi-channel runs)

    @staticmethod
    def _now():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, msg):
        line = f"[{self._now()}] {msg}"
        if not self.quiet:
            print(line, flush=True)
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    def status(self, state, extra=""):
        elapsed = (time.time() - self.t0) / 60
        best = (f"r_cut = {self.best[0]:.3f} bohr, gap = {self.best[1]:.4f} eV"
                if self.best else "-")
        text = (f"STATE        : {state}\n"
                f"started      : {self.started}\n"
                f"last update  : {self._now()}\n"
                f"elapsed      : {elapsed:.1f} min\n"
                f"aims runs    : {self.n_done} finished\n"
                f"current      : {self.current}\n"
                f"best so far  : {best}\n"
                f"host / pid   : {socket.gethostname()} / {os.getpid()}\n")
        if state == "RUNNING":
            text += ("note         : refreshed every few seconds; if 'last update' is\n"
                     "               more than ~1 min old, execute is NOT running any more\n")
        if extra:
            text += extra.rstrip() + "\n"
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(self.status_path)

    def record(self, cut, gap, seconds, state):
        self.rows.append((cut, gap, seconds, state, self.step))
        if gap is not None and (self.best is None or gap > self.best[1]):
            self.best = (cut, gap)
        with open(self.results_path, "w") as fh:
            fh.write("# r_cut(bohr)   gap(eV)   wall(min)  state    step\n")
            for c, g, s, st, sp in sorted(self.rows, key=lambda r: (r[4], r[0])):
                gs = f"{g:10.4f}" if g is not None else "       nan"
                fh.write(f"{c:12.3f} {gs} {s/60:10.1f}  {st:7s}  {sp}\n")


# ----------------------------------------------------------------- control.in

def patch_control(control_path, sep_files, out_path=None):
    """Insert/replace aimshalf_potential lines (also removes lines with the old
    keyword external_atomic_potential). sep_files:
    {species_symbol: path}."""
    text = Path(control_path).read_text()
    text = re.sub(KEYWORD_LINE, "", text, flags=re.M)
    lines = [f"{AIMS_KEYWORD} {sp} {os.path.basename(f)}"
             for sp, f in sep_files.items()]
    text = text.rstrip() + "\n\n# --- DFT-1/2 (aims_half) ---\n" + \
        "\n".join(lines) + "\n"
    Path(out_path or control_path).write_text(text)


# ----------------------------------------------------------------- aims runner

class AimsRunError(RuntimeError):
    pass


def aims_finished(path):
    """True if an aims.out exists and contains the normal-termination line."""
    p = Path(path)
    if not p.exists():
        return False
    with open(p, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 20000))
        return AIMS_DONE_MARK.encode() in fh.read()


class AimsRunner:
    """
    Runs aims and returns as soon as aims itself has finished.

    It watches aims.out instead of waiting only for the launcher: if mpirun
    stays alive more than `finish_grace` seconds after aims printed
    "Have a nice day", the launcher is terminated (it is only MPI teardown).
    timeout: seconds per aims run, 0/None = no limit.
    """

    def __init__(self, command=("mpirun", "aims.x"), output="aims.out",
                 finish_grace=30, poll=5, heartbeat=None):
        self.command = list(command)
        self.output = output
        self.finish_grace = finish_grace
        self.poll = poll
        self.heartbeat = heartbeat     # called every `poll` s while aims runs

    def _kill(self, proc):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=15)
                return
            except subprocess.TimeoutExpired:
                continue

    def run(self, cwd, timeout=None):
        out_path = Path(cwd) / self.output
        t0 = time.time()
        finished_at = None
        with open(out_path, "w") as out, open(os.devnull) as devnull:
            proc = subprocess.Popen(self.command, cwd=cwd, stdin=devnull,
                                    stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                while True:
                    try:
                        rc = proc.wait(timeout=self.poll)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if self.heartbeat:
                        self.heartbeat()
                    if finished_at is None and aims_finished(out_path):
                        finished_at = time.time()
                    if finished_at and time.time() - finished_at > self.finish_grace:
                        self._kill(proc)       # aims done, launcher stuck
                        rc = 0
                        break
                    if timeout and time.time() - t0 > timeout:
                        self._kill(proc)
                        raise AimsRunError(
                            f"aims exceeded run_timeout = {timeout} s in {cwd}")
            except BaseException:
                self._kill(proc)               # Ctrl-C / job kill: no orphans
                raise
        if not aims_finished(out_path):
            tail = out_path.read_text(errors="replace").splitlines()[-15:]
            raise AimsRunError(
                f"aims did not finish normally (exit code {rc}) in {cwd}\n"
                "last lines of aims.out:\n    " + "\n    ".join(tail))
        return AimsStdout(str(out_path))

# ----------------------------------------------------------------- fingerprints
#
# A run folder is reused only if it was made from exactly the same inputs.
# The fingerprint covers what determines the result: control.in (it already
# carries the SEP lines), geometry.in, every SEP file, and the aims binary.

FINGERPRINT_FILE = "aimshalf_fingerprint.json"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _aims_executable(command):
    """Real path of the aims binary in the launch command, or None if not found."""
    for tok in command:
        if "aims" in os.path.basename(tok):
            path = shutil.which(tok) or tok
            if os.path.isfile(path):
                return os.path.realpath(path)
    return None


def make_fingerprint(run_dir, command):
    run_dir = Path(run_dir)
    inputs = {name: _sha256(run_dir / name) for name in ("control.in", "geometry.in")}
    for sep in sorted(run_dir.glob("sep_*.dat")):
        inputs[sep.name] = _sha256(sep)
    exe = _aims_executable(command)
    exe_info = None
    if exe:
        st = os.stat(exe)
        exe_info = {"path": exe, "size": st.st_size, "mtime": int(st.st_mtime)}
    return {"inputs": inputs, "aims_executable": exe_info}


def write_fingerprint(run_dir, fp):
    (Path(run_dir) / FINGERPRINT_FILE).write_text(json.dumps(fp, indent=2) + "\n")


def read_fingerprint(run_dir):
    p = Path(run_dir) / FINGERPRINT_FILE
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def fingerprint_diff(old, new):
    """List of what changed between two fingerprints (empty = identical)."""
    diffs = []
    old_in, new_in = old.get("inputs", {}), new.get("inputs", {})
    for name in sorted(set(old_in) | set(new_in)):
        if old_in.get(name) != new_in.get(name):
            diffs.append(f"{name} changed")
    if old.get("aims_executable") != new.get("aims_executable"):
        diffs.append("aims executable changed")
    return diffs


def can_reuse(run_dir, fp, reuse_unverified=False):
    """(True, reason) if run_dir finished normally and was made from the same inputs."""
    if not aims_finished(Path(run_dir) / "aims.out"):
        return False, "aims.out missing or did not finish"
    old = read_fingerprint(run_dir)
    if old is None:
        if reuse_unverified:
            return True, "no fingerprint; reused because reuse_unverified = true"
        return False, "no fingerprint (made before aimshalf 0.2)"
    diffs = fingerprint_diff(old, fp)
    if diffs:
        return False, "; ".join(diffs)
    return True, "fingerprint matches"


def set_aside(path):
    """Rename a folder that cannot be reused to stale_<name>_<time> (never deleted)."""
    path = Path(path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new = path.with_name(f"stale_{path.name}_{stamp}")
    i = 1
    while new.exists():
        new = path.with_name(f"stale_{path.name}_{stamp}_{i}")
        i += 1
    path.rename(new)
    return new

# ----------------------------------------------------------------- SEP factory

def build_sep_file(species, corrections, r_cut, out_dir=".", xc="pb",
                   power=8, amplitude=1.0, atom_exe=None):
    """Run neutral + fractional-occupation atomic calcs, write trimmed SEP."""
    inp_n, inp_i = make_sep_inputs(species, corrections, xc=xc)
    kwargs = {} if atom_exe is None else {"exe": atom_exe}
    vt_atom = run_atom(inp_n, **kwargs)
    vt_ion = run_atom(inp_i, **kwargs)
    sep = SelfEnergyPotential(vt_atom, vt_ion)
    out = Path(out_dir) / f"sep_{species}_cut{r_cut:.3f}.dat"
    sep.write_aims_file(str(out), species, r_cut, power, amplitude)
    return str(out), sep


# ----------------------------------------------------------------- gap objective

class GapObjective:
    """
    Callable: r_cut -> band gap (eV), one aims SCF per new cut.

    - cuts are rounded to `cut_resolution` bohr; a cut already evaluated in
      this execute is never run twice
    - the inputs of a cut are first built in a hidden .staging_cut_X.XXX/
      folder and fingerprinted (control.in, geometry.in, SEP files, aims
      binary). An existing cut_X.XXX/ is reused only if it finished normally
      AND its fingerprint matches; otherwise it is renamed
      stale_cut_X.XXX_<time>/ and the cut is run again
    - a failed aims run is logged and scored as -inf; execute carries on
    """

    def __init__(self, workdir, species_corrections, runner=None, xc="pb",
                 power=8, amplitude=1.0, use_band_files=False, atom_exe=None,
                 run_timeout=None, cut_resolution=0.01, progress=None,
                 sep_builder=None, reuse_unverified=False, indirect=True):
        self.workdir = Path(workdir)
        self.species_corrections = species_corrections
        self.runner = runner or AimsRunner()
        self.xc, self.power, self.amplitude = xc, power, amplitude
        self.use_band_files = use_band_files
        self.run_timeout = run_timeout
        self.atom_exe = atom_exe
        self.cut_resolution = cut_resolution
        self.progress = progress or Progress(workdir)
        self.sep_builder = sep_builder   # callable(run_dir, r_cut) -> {species: sep file}
        self.reuse_unverified = reuse_unverified
        self.indirect = indirect          # False: smallest direct gap (minushalf default)
        self.progress.best = None
        if getattr(self.runner, "heartbeat", "absent") is None:
            self.runner.heartbeat = lambda: self.progress.status("RUNNING")
        self.cache = {}
        self.history = []  # (r_cut, gap) for successful runs

    def _tag(self):
        return f"[{self.progress.step}] " if self.progress.step else ""

    def _round(self, r_cut):
        q = self.cut_resolution
        return round(round(r_cut / q) * q, 6)

    def _gap(self, run_dir, stdout=None):
        if self.use_band_files:
            return BandStructure.from_dir(str(run_dir)).band_edges(
                indirect=self.indirect)["gap"]
        stdout = stdout or AimsStdout(str(run_dir / "aims.out"))
        return stdout.gap_ev

    def _build_inputs(self, target, r_cut):
        """control.in + geometry.in + SEP file(s) for this cut, written into target/."""
        target.mkdir()
        for f in ("control.in", "geometry.in"):
            shutil.copy(self.workdir / f, target / f)
        sep_files = {}
        if self.sep_builder is not None:
            sep_files = self.sep_builder(target, r_cut)
        for sp, corr in (self.species_corrections or {}).items():
            path, _ = build_sep_file(sp, corr, r_cut, out_dir=target,
                                     xc=self.xc, power=self.power,
                                     amplitude=self.amplitude,
                                     atom_exe=self.atom_exe)
            sep_files[sp] = path
        patch_control(target / "control.in", sep_files)

    def __call__(self, r_cut):
        r_cut = self._round(r_cut)
        if r_cut in self.cache:
            return self.cache[r_cut]
        pg = self.progress
        name = f"cut_{r_cut:.3f}"
        run_dir = self.workdir / name
        staging = self.workdir / f".staging_{name}"
        n = len(self.cache) + 1
        t0 = time.time()

        # 1. build this cut's inputs aside and fingerprint them
        if staging.exists():
            shutil.rmtree(staging)
        try:
            self._build_inputs(staging, r_cut)
        except (RuntimeError, ValueError, FileNotFoundError) as err:
            shutil.rmtree(staging, ignore_errors=True)
            pg.log(f"{self._tag()}run {n}: r_cut = {r_cut:.3f} bohr  could not build "
                   f"inputs: {err}")
            return self._store(r_cut, None, 0.0, "failed")
        fp = make_fingerprint(staging, getattr(self.runner, "command", []))

        # 2. reuse an existing run only if it was made from the same inputs
        if run_dir.exists():
            ok, why = can_reuse(run_dir, fp, self.reuse_unverified)
            if ok:
                try:
                    gap = self._gap(run_dir)
                    shutil.rmtree(staging)
                    pg.log(f"{self._tag()}run {n}: r_cut = {r_cut:.3f} bohr  reused "
                           f"{name}/ ({why})  ->  gap = {gap:.4f} eV")
                    return self._store(r_cut, gap, 0.0, "reused")
                except Exception as err:
                    why = f"results unreadable: {err}"
            old = set_aside(run_dir)
            pg.log(f"{self._tag()}{name}/ not reused ({why}); moved to {old.name}/")
        staging.rename(run_dir)
        write_fingerprint(run_dir, fp)

        # 3. run aims
        pg.current = f"aims run {n}, r_cut = {r_cut:.3f} bohr (since {pg._now()})"
        pg.status("RUNNING")
        pg.log(f"{self._tag()}run {n}: r_cut = {r_cut:.3f} bohr  started in {name}/")
        try:
            stdout = self.runner.run(str(run_dir), self.run_timeout)
            gap = self._gap(run_dir, stdout)
        except (AimsRunError, RuntimeError, ValueError, OSError) as err:
            dt = time.time() - t0
            pg.log(f"{self._tag()}run {n}: r_cut = {r_cut:.3f} bohr  FAILED after "
                   f"{dt/60:.1f} min: {err}")
            return self._store(r_cut, None, dt, "failed")
        dt = time.time() - t0
        pg.log(f"{self._tag()}run {n}: r_cut = {r_cut:.3f} bohr  finished in {dt/60:.1f} min"
               f"  ->  gap = {gap:.4f} eV")
        return self._store(r_cut, gap, dt, "ok")

    def _store(self, r_cut, gap, seconds, state):
        pg = self.progress
        pg.n_done += 1
        pg.current = "-"
        pg.record(r_cut, gap, seconds, state)
        pg.status("RUNNING")
        value = gap if gap is not None else float("-inf")
        self.cache[r_cut] = value
        if gap is not None:
            self.history.append((r_cut, gap))
        return value


def maximize_gap(objective, lo=1.0, hi=5.0, tol=0.05, max_iter=25,
                 coarse_points=0, gap_tol=0.005):
    """
    Maximize gap(r_cut) on [lo, hi] (bohr): optional coarse scan to bracket the
    peak, then golden-section search.

    Stops when ANY of these holds:
      - the bracket is narrower than `tol` bohr
      - the two interior points AND the best point differ by less than
        `gap_tol` eV (flat top: more digits are physically meaningless)
      - `max_iter` golden-section steps
    Returns (best_cut, best_gap, history, reason).
    """
    log = getattr(getattr(objective, "progress", None), "log", print)
    lo0, hi0 = lo, hi                     # full search range, for the edge check
    if coarse_points >= 3:
        import numpy as np
        grid = np.linspace(lo, hi, coarse_points)
        log(f"coarse scan: {coarse_points} cuts from {lo} to {hi} bohr")
        vals = [objective(x) for x in grid]
        if all(v == float("-inf") for v in vals):
            raise RuntimeError("every aims run of the coarse scan failed; "
                               "see aimshalf_execute.log")
        i = int(np.argmax(vals))
        lo = grid[max(i - 1, 0)]
        hi = grid[min(i + 1, len(grid) - 1)]
        log(f"coarse scan done: peak bracketed in [{lo:.3f}, {hi:.3f}] bohr")
    a, b = lo, hi
    c = b - GOLDEN * (b - a)
    d = a + GOLDEN * (b - a)
    fc, fd = objective(c), objective(d)
    it = 0
    reason = f"max_iter = {max_iter} reached"
    while True:
        best_gap = max((g for _, g in objective.history), default=float("-inf"))
        if abs(b - a) <= tol:
            reason = f"bracket narrower than tol = {tol} bohr"
            break
        if (abs(fc - fd) < gap_tol and abs(max(fc, fd) - best_gap) < gap_tol
                and it > 0):
            reason = f"flat top: gaps agree within gap_tol = {gap_tol} eV"
            break
        if it >= max_iter:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = objective(d)
        it += 1
    if not objective.history:
        raise RuntimeError("no aims run succeeded; see aimshalf_execute.log")
    best = max(objective.history, key=lambda t: t[1])
    if best[0] - lo0 < 2 * tol or hi0 - best[0] < 2 * tol:
        log(f"WARNING: optimal r_cut = {best[0]:.3f} bohr is at the edge of the search "
            f"range [{lo0}, {hi0}] bohr; the true maximum may lie outside. Widen "
            f"cut_min/cut_max (or cut_ranges) and run again.")
        reason += "; WARNING: optimum at the edge of the search range"
    return best[0], best[1], objective.history, reason


# =============================================================================
# Automatic correction selection + minushalf-style channel-by-channel cuts
# =============================================================================
#
# A "channel" is one (species, l, dq): e.g. ("N", 1, 0.5). Like minushalf, every
# channel gets its OWN atomic calculation and its OWN r_cut; channels are
# optimized one after another (each later step keeps the earlier cuts fixed)
# and the trimmed SEPs of all channels of a species are summed into one file.

L_NAME = "spdfgh"
L_INDEX = {c: i for i, c in enumerate(L_NAME)}

_ATOM_CACHE = {}


def _vtotals(species, l, dq, xc, atom_exe=None):
    """(neutral, ion) VTOTALs for one channel, cached for the whole execute."""
    key = (species, l, round(dq, 6), xc)
    if key not in _ATOM_CACHE:
        inp_n, inp_i = make_sep_inputs(species, {l: dq}, xc=xc)
        kw = {} if atom_exe is None else {"exe": atom_exe}
        _ATOM_CACHE[key] = (run_atom(inp_n, **kw), run_atom(inp_i, **kw))
    return _ATOM_CACHE[key]


VALID_CODES = ("v", "vf", "c", "cf", "vc", "vfc", "vcf", "vfcf")


def parse_correction_code(code):
    """minushalf correction codes -> (valence, valence fractional, conduction,
    conduction fractional). 'v' = valence, 'c' = conduction, 'f' after a letter =
    fractional (every orbital above the threshold) instead of the dominant one."""
    if code not in VALID_CODES:
        raise ValueError(f"correction_code '{code}' not valid; use one of {', '.join(VALID_CODES)}")
    return "v" in code, "vf" in code, "c" in code, "cf" in code


def channels_from_corrections(corrections, conduction=False):
    """{'N': {'p': 0.5}} -> [{'species':'N','l':1,'dq':0.5,'conduction':False}, ...]"""
    out = []
    for sp, d in (corrections or {}).items():
        for l, dq in d.items():
            l_int = L_INDEX[l] if isinstance(l, str) else int(l)
            out.append({"species": sp, "l": l_int, "dq": float(dq), "conduction": conduction})
    return out


def channel_label(ch):
    """'As-p' for a valence channel, 'Ga-s-cond' for a conduction channel."""
    return f"{ch['species']}-{L_NAME[ch['l']]}" + ("-cond" if ch.get("conduction") else "")


def write_species_sep(run_dir, species, chans, xc="pb", power=8, amplitude=1.0,
                      atom_exe=None):
    """Sum the trimmed SEPs of all channels (each with its own 'cut') of one
    species and write the file aims reads. Returns the file path.
    A conduction channel enters with amplitude -|A| (repulsive: pushes the
    conduction band up), as minushalf does; a valence channel with +A."""
    radius, total, desc = None, None, []
    for ch in chans:
        cond = bool(ch.get("conduction"))
        amp = -abs(amplitude) if cond else amplitude
        vt_n, vt_i = _vtotals(species, ch["l"], ch["dq"], xc, atom_exe)
        sep = SelfEnergyPotential(vt_n, vt_i)
        v = sep.trimmed(ch["cut"], power, amp)
        if radius is None:
            radius, total = sep.radius, v.copy()
        else:
            total += np.interp(radius, sep.radius, v)
        desc.append(f"{L_NAME[ch['l']]}: dq={ch['dq']:.4f} r_cut={ch['cut']:.3f}"
                    + (" conduction (A<0)" if cond else ""))
    tag = lambda c: ("c" if c.get("conduction") else "") + L_NAME[c["l"]]
    if len(chans) == 1:
        c0 = chans[0]
        name = f"sep_{species}_{'c' if c0.get('conduction') else ''}cut{c0['cut']:.3f}.dat"
    else:
        name = f"sep_{species}_" + "_".join(f"{tag(c)}{c['cut']:.3f}" for c in chans) + ".dat"
    path = Path(run_dir) / name
    rmax = max(c["cut"] for c in chans)
    mask = radius >= 1e-4
    r_out, v_out = radius[mask], total[mask]
    stop = np.searchsorted(r_out, rmax * 1.10)
    r_out, v_out = r_out[:stop + 1], v_out[:stop + 1]
    with open(path, "w") as fh:
        fh.write("# aims_half trimmed self-energy potential (sum of channels)\n")
        fh.write(f"# species: {species}\n")
        fh.write(f"# channels: {' | '.join(desc)}  power: {power}  amplitude: {amplitude:.4f}\n")
        fh.write("# columns: r(bohr)  V_S(Hartree)\n")
        fh.write(f"{len(r_out)}\n")
        for r, vv in zip(r_out, v_out):
            fh.write(f"{r:22.14e} {vv:22.14e}\n")
    return str(path)


def make_channel_builder(fixed, active, xc="pb", power=8, amplitude=1.0,
                         atom_exe=None):
    """sep_builder for GapObjective: fixed channels keep their cut, the active
    channel takes the trial r_cut."""
    def build(run_dir, r_cut):
        chans = [dict(c) for c in fixed] + [dict(active, cut=r_cut)]
        by_sp = {}
        for c in chans:
            by_sp.setdefault(c["species"], []).append(c)
        return {sp: write_species_sep(run_dir, sp, cs, xc, power, amplitude, atom_exe)
                for sp, cs in by_sp.items()}
    return build


# ----------------------------------------------------------------- PBE reference

def run_pbe_reference(workdir, runner, ref_dir="pbe_ref", progress=None, timeout=None,
                      reuse_unverified=False):
    """Plain DFT run with Mulliken-resolved bands (no SEP) in workdir/ref_dir.
    Every `output band` line of control.in becomes `output band_mulliken`
    (which also writes the normal band files). An existing ref_dir is reused
    only if it finished and its fingerprint matches; otherwise it is renamed
    stale_<ref_dir>_<time>/ and the reference is run again."""
    pg = progress
    d = Path(workdir) / ref_dir
    text = (Path(workdir) / "control.in").read_text()
    text = re.sub(KEYWORD_LINE, "", text, flags=re.M)
    text, n = re.subn(r"^(\s*output\s+)band(\s)", r"\1band_mulliken\2", text, flags=re.M)
    if n == 0:
        raise RuntimeError("automatic corrections need 'output band ...' lines in "
                           "control.in (the band path is used to find the VBM character)")
    staging = Path(workdir) / f".staging_{ref_dir}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    (staging / "control.in").write_text(text)
    shutil.copy(Path(workdir) / "geometry.in", staging / "geometry.in")
    fp = make_fingerprint(staging, getattr(runner, "command", []))

    if d.exists():
        ok, why = can_reuse(d, fp, reuse_unverified)
        if ok and glob.glob(str(d / "bandmlk*.out")):
            shutil.rmtree(staging)
            if pg:
                pg.log(f"PBE reference: reusing {ref_dir}/ ({why})")
            return d
        if ok:
            why = "no bandmlk*.out files"
        old = set_aside(d)
        if pg:
            pg.log(f"PBE reference: {ref_dir}/ not reused ({why}); moved to {old.name}/")
    staging.rename(d)
    write_fingerprint(d, fp)

    if pg:
        pg.current = f"PBE reference run with Mulliken bands in {ref_dir}/ (since {pg._now()})"
        pg.status("RUNNING")
        pg.log(f"PBE reference: running plain DFT with band_mulliken in {ref_dir}/")
    t0 = time.time()
    runner.run(str(d), timeout)
    if pg:
        pg.log(f"PBE reference: finished in {(time.time() - t0)/60:.1f} min")
        pg.current = "-"
    return d


def select_channels(ref_dir, code="v", threshold=10.0, progress=None, xc="pb",
                    indirect=True, overwrite_vbm=None, overwrite_cbm=None,
                    vbm_characters=None, divide_character=None,
                    conduction_threshold=9.0, cbm_characters=None):
    """
    minushalf rules on the band-edge characters of the PBE reference.

    correction_code (minushalf): 'v' valence, 'c' conduction, 'f' fractional:
      v / c   : the single (species, l) with the largest share of the VBM / CBM
      vf / cf : every (species, l) above `threshold` / `conduction_threshold` %,
                each with round(100 * p / sum p) percent of the half electron
      vc, vfc, vcf, vfcf: valence channels first, then conduction channels
    Valence channels remove charge and lower the VBM (+A); conduction channels use
    the same atomic calculation with -|A| and raise the CBM, as in minushalf.
    The edges follow minushalf: `indirect` (False = at the k-point of the smallest
    direct gap), `overwrite_vbm` / `overwrite_cbm` = [k, state] to pick the band by
    hand, `vbm_characters` / `cbm_characters` = [[species, l, percent], ...] to
    replace character values.
    A channel whose atom has n nearest neighbours of the same species (geometry.in;
    `divide_character` = [[species, l, n], ...] overrides n) gets 1/(1+n) of its
    share, as minushalf does (Si: 1/4 e).
    Channels whose orbital is empty in the free atom (e.g. Al-d) cannot lose
    charge; they are dropped with a warning and the rest renormalized.
    Writes ref_dir/vbm_character.txt. Returns (channels, report_text).
    """
    from .aims_io import (edge_character, character_table, species_character, atoms_map,
                          equal_neighbors)
    from .atom_runner import AtomInp
    do_v, frac_v, do_c, frac_c = parse_correction_code(code)
    ref = Path(ref_dir)
    files = sorted(glob.glob(str(ref / "bandmlk*.out")))
    if not files:
        raise RuntimeError(f"no bandmlk*.out files in {ref}")
    amap = atoms_map(str(ref / "geometry.in"))
    lines = [f"band edges: {'fundamental gap (indirect: true)' if indirect else 'smallest direct gap (indirect: false)'}"]
    sc_all = {}
    for which, where, over in (("vbm", overwrite_vbm, vbm_characters),
                               ("cbm", overwrite_cbm, cbm_characters)):
        info, proj = edge_character(files, which, indirect=indirect, location=where or None)
        sc = species_character(character_table(proj, amap))
        lines.append(f"{which.upper()}: E = {info['energy']:.4f} eV at k-point {info['k_number']} = ("
                     f"{', '.join(f'{x:.4f}' for x in info['kpoint'])}), "
                     f"{len(info['states'])} degenerate state(s) {info['states']}"
                     + (" [set by overwrite]" if where else ""))
        for (sp, l), p in sc.items():
            if p >= 0.5:
                lines.append(f"    {sp:>3s}-{l}  {p:6.2f} %")
        for sp, l, pct in (over or []):
            sc[(str(sp).capitalize(), str(l).lower())] = float(pct)
            lines.append(f"    {which}_characters: {str(sp).capitalize()}-{str(l).lower()} "
                         f"set to {float(pct):.2f} %")
        sc_all[which] = dict(sorted(sc.items(), key=lambda kv: -kv[1]))

    def has_electrons(sp, l):
        inp = AtomInp.for_species(sp)
        return any(o["l"] == L_INDEX[l] and o["occ_down"] > 1e-6 for o in inp.orbitals)

    forced = {(str(sp).capitalize(), str(l).lower()): int(n) for sp, l, n in (divide_character or [])}

    def choose(which, fractional, thr):
        sc = sc_all[which]
        if fractional:
            picked = {k: p for k, p in sc.items() if p > thr}
        else:
            (sp, l), p = next(iter(sc.items()))
            picked = {(sp, l): p}
        for k in [k for k in picked if not has_electrons(*k)]:
            picked.pop(k)
            lines.append(f"    dropped {k[0]}-{k[1]} ({which.upper()}): orbital empty in the free atom")
        if not picked:
            raise RuntimeError(f"no correctable channel found at the {which.upper()}; "
                               "see vbm_character.txt")
        s = sum(picked.values())
        out = []
        for (sp, l), p in sorted(picked.items(), key=lambda kv: -kv[1]):
            n = forced.get((sp, l))
            if n is None:
                n = equal_neighbors(str(ref / "geometry.in"), sp)
            if n:
                lines.append(f"    {sp}-{l}: {n} nearest neighbour(s) of the same species"
                             f"{' (divide_character)' if (sp, l) in forced else ''}"
                             f" -> 1/{1 + n} of its share")
            pct = round((100 / (1 + n)) * p / s)
            out.append({"species": sp, "l": L_INDEX[l], "dq": 0.5 * pct / 100,
                        "conduction": which == "cbm"})
        return out

    lines.append(f"correction_code = {code}"
                 + (f", valence threshold = {threshold} %" if frac_v else "")
                 + (f", conduction threshold = {conduction_threshold} %" if frac_c else ""))
    channels = []
    if do_v:
        channels += choose("vbm", frac_v, threshold)
    if do_c:
        channels += choose("cbm", frac_c, conduction_threshold)
    lines.append("selected channels: " + ", ".join(
        f"{channel_label(c)} (dq = {c['dq']:.4f} e)" for c in channels))
    text = "\n".join(lines)
    (ref / "vbm_character.txt").write_text(text + "\n")
    if progress:
        for ln in lines:
            progress.log(ln)
    return channels, text
