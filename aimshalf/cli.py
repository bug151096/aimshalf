"""
aimshalf command-line interface.

Commands
--------
gen-sep      Generate a trimmed SEP file for one species.
band-gap     Report VBM/CBM/gap from band*.out files in a directory.
vbm-character / cbm-character / band-character
             Per-atom, per-orbital Mulliken character of the band edge or of
             any (k-point, state) (requires bandmlk files; check with inspect-mlk).
inspect-mlk  Show how a bandmlk file is being interpreted (validation aid).
execute      Full correction loop: optimize r_cut maximizing the gap.
"""
import argparse
import json
import sys
from pathlib import Path


def cmd_gen_sep(args):
    """One SEP file. Several -c options = several channels, each from its own
    atomic calculation, summed (the same scheme `execute` uses), all at --cut."""
    from .workflow import write_species_sep, _vtotals, L_INDEX
    from .sep import SelfEnergyPotential
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chans = []
    for item in args.correction:
        l, dq = item.split("=")
        chans.append({"species": args.species, "l": L_INDEX[l.strip()],
                      "dq": float(dq), "cut": args.cut})
    path = write_species_sep(out_dir, args.species, chans, xc=args.xc,
                             power=args.power, amplitude=args.amplitude)
    print(f"SEP written: {path}")
    qs = [SelfEnergyPotential(*_vtotals(args.species, c["l"], c["dq"], args.xc))
          .removed_charge_estimate() for c in chans]
    if all(q is not None for q in qs):
        print(f"tail-charge check: fitted q = {sum(qs):.4f} e "
              f"(expected ~ {sum(c['dq'] for c in chans):.4f})")


def cmd_band_gap(args):
    """Gap from band*.out. Default: smallest direct gap (minushalf); -i: fundamental."""
    from .aims_io import BandStructure
    bs = BandStructure.from_dir(args.dir)
    e = bs.band_edges(indirect=args.indirect)

    def where(k, b):
        kx = ", ".join(f"{x:.4f}" for x in bs.kpoints[k])
        return f"k-point {k + 1} = ({kx}), state {b + 1}"
    print(f"VBM: {e['vbm']:.4f} eV at {where(e['vbm_k'], e['vbm_band'])}")
    print(f"CBM: {e['cbm']:.4f} eV at {where(e['cbm_k'], e['cbm_band'])}")
    kind = "direct" if e["direct"] else "indirect"
    mode = "fundamental gap (-i)" if args.indirect else "smallest direct gap"
    print(f"gap: {e['gap']:.4f} eV ({kind})   [{mode}]")


def _character(args, which, location=None):
    """Orbital character of a band edge (or of a given k-point and state) from the
    bandmlk files, summed over the states degenerate with it (same routine as the
    automatic mode of execute)."""
    import glob
    from .aims_io import edge_character, character_table, species_character, atoms_map
    files = args.mlk_files or sorted(glob.glob(str(Path(args.dir) / "bandmlk*.out")))
    if not files:
        sys.exit(f"no bandmlk*.out files in {args.dir}; run aims with "
                 "'output band_mulliken ...' lines in control.in")
    try:
        info, proj = edge_character(files, which, indirect=getattr(args, "indirect", True),
                                    location=location)
    except ValueError as err:
        sys.exit(str(err))
    rows = character_table(proj, atoms_map(str(Path(args.dir) / "geometry.in")))
    k = ", ".join(f"{x:.4f}" for x in info["kpoint"])
    label = "BAND" if location else which.upper()
    print(f"{label}: E = {info['energy']:.4f} eV at k-point {info['k_number']} = ({k}), "
          f"states {info['states']} summed (degenerate), file {Path(info['file']).name}")
    print("\nby species and orbital:")
    for (sp, l), p in species_character(rows).items():
        if p >= 0.5:
            print(f"  {sp:>3s}-{l}  {p:6.2f} %")
    print("\nby atom:")
    print(f"{'atom':>6} {'species':>8} {'l':>3} {'percent':>9}")
    for r in rows:
        if r["percent"] >= 0.5:
            print(f"{r['atom']:>6} {r['species']:>8} {r['l']:>3} {r['percent']:>8.2f}%")


def cmd_vbm(args):
    _character(args, "vbm")


def cmd_cbm(args):
    _character(args, "cbm")


def cmd_band(args):
    _character(args, "vbm", location=(args.kpoint, args.band))


def cmd_inspect_mlk(args):
    from .aims_io import MullikenBandFile
    mlk = MullikenBandFile(args.mlk_file)
    k0 = mlk.blocks[0]
    s = sorted(k0)[0]
    atom = sorted(k0[s]["atoms"])[0]
    print(f"file           : {args.mlk_file}")
    print(f"k-point blocks : {len(mlk.blocks)}")
    print(f"states per k   : {len(k0)}")
    print(f"spin column    : {'yes (SOC or spin-polarized)' if mlk.has_spin_column else 'no'}")
    print(f"sample         : k-point 1, state {s}, E = {k0[s]['eig']:.4f} eV, "
          f"occ = {k0[s]['occ']:.4f}")
    print(f"                 atom {atom}: {json.dumps(k0[s]['atoms'][atom])}")
    print("Compare the sample with the same line of the file. The orbital columns "
          "after 'total' are read as s, p, d, f, ... (MullikenBandFile.L_LABELS).")


def _execute_already_running(directory):
    """PID of a live execute in `directory` (from its status file), else None."""
    import os, socket
    st = Path(directory) / "aimshalf_status.txt"
    if not st.exists():
        return None
    info = {k.strip(): v.strip() for k, v in
            (l.split(":", 1) for l in st.read_text().splitlines() if ":" in l)}
    if info.get("STATE", "").strip() != "RUNNING":
        return None
    try:
        host, pid = [x.strip() for x in info["host / pid"].split("/")]
        pid = int(pid)
    except (KeyError, ValueError):
        return None
    if host != socket.gethostname() or pid == os.getpid():
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _detach_execute(directory):
    """Relaunch this same command in the background, immune to closing the
    terminal (own session = what nohup gives). Output -> execute.out."""
    import os, subprocess
    env = dict(os.environ, AIMSHALF_DETACHED="1")
    out_path = Path(directory) / "execute.out"
    with open(out_path, "a") as out, open(os.devnull) as devnull:
        proc = subprocess.Popen([sys.executable, "-m", "aimshalf.cli"] + sys.argv[1:],
                                stdin=devnull, stdout=out, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
    print(f"aimshalf execute is running in the background (PID {proc.pid}).")
    print("You can close this terminal. Check progress with:")
    print("    cat aimshalf_status.txt        # RUNNING / DONE / FAILED")
    print("    tail -f aimshalf_execute.log   # live log (Ctrl-C only stops watching)")
    print(f"Stop it with:  kill {proc.pid}    (run 'aimshalf execute' again later to resume)")
    

def _binary_contains(path, needle, chunk=1 << 22):
    """True if the file contains the byte string (e.g. a keyword compiled into aims)."""
    tail = b""
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                return False
            data = tail + block
            if needle in data:
                return True
            tail = data[-(len(needle) - 1):]


def _preflight(directory, corrections, auto, aims_cmd, use_band_files, xc,
               indirect=True, code="v", conduction_corrections=None):
    """Problems that would make every aims run fail, checked before execute starts
    (and before it goes to the background). Returns (errors, warnings)."""
    import re
    from .atom_runner import DEFAULT_EXE, FIXTURES
    from .workflow import _aims_executable, L_INDEX, VALID_CODES, AIMS_KEYWORD
    d = Path(directory)
    if not d.is_dir():
        return [f"folder not found: {d}"], []
    errors, warnings = [], []
    for f in ("control.in", "geometry.in"):
        if not (d / f).is_file():
            errors.append(f"{f} not found in {d.resolve()}")
    if not DEFAULT_EXE.exists():
        errors.append(f"atomic solver not built: {DEFAULT_EXE} "
                      "(run 'make atom' in the aimshalf source folder)")
    exe = _aims_executable(aims_cmd.split())
    if exe is None:
        errors.append(f"no aims executable found in aims_cmd = '{aims_cmd}' "
                      "(set aims_cmd in aimshalf.yaml to the patched aims binary)")
    elif not _binary_contains(exe, AIMS_KEYWORD.encode()):
        errors.append(f"{exe} does not know the keyword '{AIMS_KEYWORD}': rebuild FHI-aims "
                      "with the current hooks (python3 fhiaims/install_hooks.py <aims dir>)")
    control = (d / "control.in").read_text() if (d / "control.in").is_file() else ""
    if control and (use_band_files or auto) and not re.search(
            r"^\s*output\s+band\s", control, re.M):
        errors.append("control.in has no 'output band ...' lines (needed for the "
                      "band-path gap and for the automatic mode)")
    if not indirect and not use_band_files:
        errors.append("indirect: false (smallest direct gap, the minushalf default) needs "
                      "use_band_files: true; or set indirect: true")
    m = re.search(r"^\s*xc\s+(\S+)", control, re.M)
    if m and xc == "pb" and m.group(1).lower() != "pbe":
        warnings.append(f"control.in uses xc '{m.group(1)}' but the atomic calculations "
                        "use PBE (xc: pb); the two should match")
    if auto and code not in VALID_CODES:
        errors.append(f"correction_code '{code}' not valid; use one of {', '.join(VALID_CODES)}")
    if auto and conduction_corrections:
        errors.append("conduction_corrections needs explicit `corrections`; in automatic "
                      "mode use a correction_code with 'c' (e.g. vc, vfcf)")
    explicit = {} if auto else dict(corrections)
    for sp, chans in (conduction_corrections or {}).items():
        explicit.setdefault(sp, {}).update(chans)
    if explicit:
        for sp, chans in explicit.items():
            if not (FIXTURES / sp / "INP").is_file():
                errors.append(f"no atomic input for species '{sp}' ({FIXTURES / sp / 'INP'})")
            for l in chans:
                if isinstance(l, str) and l not in L_INDEX:
                    errors.append(f"unknown orbital '{l}' for {sp} (use s, p, d, f)")
    return errors, warnings


def cmd_execute(args):
    import os
    import shutil
    import time
    from .workflow import (GapObjective, maximize_gap, AimsRunner, Progress,
                           channels_from_corrections, channel_label,
                           make_channel_builder, run_pbe_reference, select_channels)
    from .config import load_config

    cfg, path = load_config(getattr(args, "config", None), start_dir=args.dir or ".")

    def pick(cli_val, key):
        return cli_val if cli_val is not None else cfg[key]

    corrections = json.loads(args.corrections) if args.corrections is not None \
        else cfg["corrections"]
    auto = corrections in (None, "auto", {}, "")

    directory      = pick(args.dir, "dir")
    aims_cmd       = pick(args.aims_cmd, "aims_cmd")
    xc             = pick(args.xc, "xc")
    cut_min        = pick(args.cut_min, "cut_min")
    cut_max        = pick(args.cut_max, "cut_max")
    tol            = pick(args.tol, "tol")
    coarse_points  = pick(args.coarse_points, "coarse_points")
    use_band_files = pick(args.use_band_files, "use_band_files")
    cut_ranges     = cfg["cut_ranges"] or {}

    errors, warnings = _preflight(directory, corrections, auto, aims_cmd,
                                  use_band_files, xc, cfg["indirect"],
                                  cfg["correction_code"], cfg["conduction_corrections"])
    for w in warnings:
        print(f"warning: {w}")
    if errors:
        sys.exit("aimshalf execute cannot start:\n  - " + "\n  - ".join(errors))

    running = _execute_already_running(directory)
    if running:
        sys.exit(f"an aimshalf execute is already running in this folder (PID {running}); "
                 "see aimshalf_status.txt. Stop it with 'kill " + str(running) + "' first.")

    in_batch_job = any(os.environ.get(v) for v in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID"))
    if not (args.foreground or in_batch_job or os.environ.get("AIMSHALF_DETACHED")):
        _detach_execute(directory)
        return

    pg = Progress(directory)
    pg.log("=" * 70)
    pg.log(f"aimshalf execute started in {Path(directory).resolve()}")
    pg.log(f"config: {path or 'none (defaults + command line)'}")
    pg.log(f"corrections: {'AUTO (' + cfg['correction_code'] + ')' if auto else corrections}"
           + (f"  conduction: {cfg['conduction_corrections']}"
              if not auto and cfg["conduction_corrections"] else "")
           + f"   xc: {xc}")
    pg.log(f"aims command: {aims_cmd}")
    pg.log("gap: " + ("fundamental (indirect: true)" if cfg["indirect"]
                      else "smallest direct gap (indirect: false, minushalf default)"))
    pg.log(f"r_cut range [{cut_min}, {cut_max}] bohr"
           + (f" (per species: {cut_ranges})" if cut_ranges else "")
           + f", coarse points {coarse_points}, tol {tol} bohr, gap_tol {cfg['gap_tol']} eV")
    pg.log("watch progress:  cat aimshalf_status.txt   |   tail -f aimshalf_execute.log")
    pg.status("RUNNING")

    runner = AimsRunner(aims_cmd.split(), finish_grace=cfg["finish_grace"])
    timeout = cfg["run_timeout"] or None
    try:
        if auto:
            ref = run_pbe_reference(directory, runner, cfg["pbe_ref_dir"], pg, timeout,
                                    reuse_unverified=cfg["reuse_unverified"])
            channels, _ = select_channels(ref, cfg["correction_code"],
                                          cfg["valence_threshold"], pg, xc,
                                          indirect=cfg["indirect"],
                                          overwrite_vbm=cfg["overwrite_vbm"],
                                          overwrite_cbm=cfg["overwrite_cbm"],
                                          vbm_characters=cfg["vbm_characters"],
                                          divide_character=cfg["divide_character"],
                                          conduction_threshold=cfg["conduction_threshold"],
                                          cbm_characters=cfg["cbm_characters"])
        else:
            channels = (channels_from_corrections(corrections) +
                        channels_from_corrections(cfg["conduction_corrections"], conduction=True))

        multi = len(channels) > 1
        if multi:
            pg.log(f"{len(channels)} channels, optimized one after another (minushalf "
                   f"scheme): " + ", ".join(channel_label(c) for c in channels))
        fixed, results = [], []
        for i, ch in enumerate(channels, 1):
            lab = channel_label(ch)
            lo, hi = cut_ranges.get(ch["species"], (cut_min, cut_max))
            step_dir = (Path(directory) / f"step{i}_{lab}") if multi else Path(directory)
            if multi:
                step_dir.mkdir(exist_ok=True)
                for f in ("control.in", "geometry.in"):
                    shutil.copy(Path(directory) / f, step_dir / f)
                pg.log(f"--- step {i}/{len(channels)}: {lab}, dq = {ch['dq']:.4f} e, "
                       f"r_cut in [{lo}, {hi}] bohr" +
                       (f"; fixed: " + ", ".join(f"{channel_label(c)} @ {c['cut']:.3f}"
                                                 for c in fixed) if fixed else ""))
            pg.step = f"{i}/{len(channels)} {lab}" if multi else ""
            obj = GapObjective(step_dir, None, runner=runner, xc=xc,
                               use_band_files=use_band_files,
                               run_timeout=timeout,
                               cut_resolution=cfg["cut_resolution"], progress=pg,
                               reuse_unverified=cfg["reuse_unverified"],
                               indirect=cfg["indirect"],
                               sep_builder=make_channel_builder(
                                   fixed, ch, xc, cfg["power"], cfg["amplitude"]))
            cut, gap, history, reason = maximize_gap(
                obj, lo, hi, tol=tol, coarse_points=coarse_points,
                gap_tol=cfg["gap_tol"], max_iter=cfg["max_iter"])
            pg.log(f"{lab}: optimal r_cut = {cut:.3f} bohr, gap = {gap:.4f} eV ({reason})")
            fixed.append(dict(ch, cut=cut))
            results.append((lab, ch["dq"], cut, gap))
    except KeyboardInterrupt:
        pg.log("interrupted by user (Ctrl-C / job killed)")
        pg.status("FAILED", "reason       : interrupted; run the same command again to "
                  "resume (finished runs are reused)")
        sys.exit(130)
    except Exception as err:
        pg.log(f"execute FAILED: {err}")
        pg.status("FAILED", f"reason       : {err}")
        raise

    summary = "  +  ".join(f"{lab} (dq {dq:.3f}) r_cut = {cut:.3f}" for lab, dq, cut, _ in results)
    final_gap = results[-1][3]
    pg.log(f"RESULT: {summary}  ->  gap = {final_gap:.4f} eV")
    pg.status("DONE", f"RESULT       : {summary}\n"
                      f"final gap    : {final_gap:.4f} eV")


def main(argv=None):
    from . import __version__
    ap = argparse.ArgumentParser(
        prog="aimshalf",
        description="DFT-1/2 (LDA-1/2 / GGA-1/2) band-gap correction for FHI-aims. "
                    "Run 'aimshalf <command> -h' for the options of a command.")
    ap.add_argument("--version", action="version", version=f"aimshalf {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<command>")

    p = sub.add_parser(
        "execute", help="full DFT-1/2 run: optimize r_cut by maximizing the gap",
        description="Run the DFT-1/2 correction in a folder with control.in, geometry.in "
                    "and (optionally) aimshalf.yaml. Without explicit corrections, a plain "
                    "PBE run in pbe_ref/ decides which orbitals to correct from the VBM "
                    "Mulliken character. Runs in the background unless --foreground or "
                    "inside a SLURM/PBS/LSF job; watch aimshalf_status.txt.")
    p.add_argument("--config", default=None,
                   help="YAML config (default: aimshalf.yaml in --dir)")
    p.add_argument("--dir", default=None, help="calculation folder (default: current)")
    p.add_argument("--corrections", default=None,
                   help='JSON, e.g. \'{"N": {"p": 0.5}}\' ; omit (or "auto") for '
                        'automatic selection from a PBE Mulliken run')
    p.add_argument("--aims-cmd", default=None, help='e.g. "mpirun -np 8 aims.x"')
    p.add_argument("--xc", default=None, help="ATOM functional code (pb = PBE, ca = LDA)")
    p.add_argument("--cut-min", type=float, default=None, help="bohr")
    p.add_argument("--cut-max", type=float, default=None, help="bohr")
    p.add_argument("--tol", type=float, default=None, help="r_cut bracket tolerance (bohr)")
    p.add_argument("--coarse-points", type=int, default=None)
    p.add_argument("--use-band-files", action="store_true", default=None,
                   help="gap from band*.out (band path) instead of the aims.out estimate")
    p.add_argument("--foreground", action="store_true",
                   help="stay attached to the terminal (default: run in background); "
                        "automatic inside SLURM/PBS/LSF batch jobs")
    p.set_defaults(func=cmd_execute)

    p = sub.add_parser(
        "gen-sep", help="write one trimmed self-energy potential file",
        description="Atomic calculations + trimmed SEP for one species at a given r_cut. "
                    "Several -c options = several channels (one atomic run each), summed.")
    p.add_argument("species")
    p.add_argument("--correction", "-c", action="append", required=True,
                   metavar="l=dq", help="e.g. -c p=0.5 (repeatable)")
    p.add_argument("--cut", type=float, required=True, help="r_cut in bohr")
    p.add_argument("--xc", default="pb", help="ATOM functional code (pb = PBE)")
    p.add_argument("--power", type=int, default=8, help="trimming exponent")
    p.add_argument("--amplitude", type=float, default=1.0,
                   help="-1.0 for a conduction (+1/2) correction")
    p.add_argument("--out-dir", default=".")
    p.set_defaults(func=cmd_gen_sep)

    p = sub.add_parser("band-gap", help="VBM, CBM and gap from the band*.out files of a folder",
                       description="Default: smallest direct gap (minushalf default); "
                                   "-i: fundamental gap, VBM and CBM at any k-point.")
    p.add_argument("--dir", default=".")
    p.add_argument("-i", "--indirect", action="store_true", help="fundamental (indirect) gap")
    p.set_defaults(func=cmd_band_gap)

    for name, fn, what in (("vbm-character", cmd_vbm, "valence band maximum"),
                           ("cbm-character", cmd_cbm, "conduction band minimum")):
        p = sub.add_parser(
            name, help=f"Mulliken orbital character of the {what}",
            description=f"Orbital character of the {what}, summed over degenerate states, "
                        "from the bandmlk*.out files (aims: 'output band_mulliken ...').")
        p.add_argument("mlk_files", nargs="*",
                       help="bandmlk*.out files (default: all of them in --dir)")
        p.add_argument("--dir", default=".",
                       help="calculation folder (geometry.in, bandmlk*.out)")
        p.add_argument("-i", "--indirect", action="store_true",
                       help="edge of the fundamental gap (default: at the k-point of the "
                            "smallest direct gap, as minushalf)")
        p.set_defaults(func=fn)

    p = sub.add_parser(
        "band-character", help="Mulliken orbital character of any (k-point, state)",
        description="Character of state BAND at k-point KPOINT (both numbered from 1; "
                    "k-points counted over all bandmlk files in order, as band-gap prints "
                    "them), summed over states degenerate with it.")
    p.add_argument("kpoint", type=int)
    p.add_argument("band", type=int)
    p.add_argument("mlk_files", nargs="*",
                   help="bandmlk*.out files (default: all of them in --dir)")
    p.add_argument("--dir", default=".", help="calculation folder (geometry.in, bandmlk*.out)")
    p.set_defaults(func=cmd_band)

    p = sub.add_parser("inspect-mlk", help="show how a bandmlk file is read (format check)")
    p.add_argument("mlk_file")
    p.set_defaults(func=cmd_inspect_mlk)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
