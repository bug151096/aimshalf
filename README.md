# aimshalf — DFT-1/2 for FHI-aims

aimshalf applies the DFT-1/2 band-gap correction (LDA-1/2 / GGA-1/2; Ferreira, Marques &
Teles, 2008) to the all-electron code [FHI-aims](https://fhi-aims.org). It follows the
workflow of [minushalf](https://minushalf.readthedocs.io), the GMSN/ITA implementation for
VASP, and reuses its atomic solver and atomic inputs: the self-energy potentials it builds
are numerically identical to minushalf's (see *Validation*).

## How it works

1. **Atomic calculations** (L. G. Ferreira's modified ATOM, bundled): the neutral atom and
   the atom with a fraction of an electron removed from the orbital that hosts the hole.
2. **Self-energy potential**: V_S(r) = [V_ion(r) − V_atom(r)] · Θ(r), with
   Θ(r) = A [1 − (r/r_cut)^8]^3 for r < r_cut and 0 beyond.
3. **FHI-aims** (patched) adds V_S around every atom of the corrected species to the
   Kohn–Sham potential at every SCF iteration.
4. **r_cut is variational**: the value that maximizes the band gap, found by a coarse scan
   followed by a golden-section search.

All DFT-1/2 physics lives in Python. On the FHI-aims side there is a small, generic hook (an
"external atom-centred local potential", keyword `aimshalf_potential` in control.in)
in five source files.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .          # aimshalf command + numpy, pyyaml
make atom                 # builds the atomic solver atom/atmhalf (needs gfortran)
pip install -e ".[dev]" && python -m pytest tests/   # pytest + matplotlib (scripts, validation)
```

Install in editable mode (`-e`): the atomic solver and its inputs are read from the
source folder (or from `$AIMSHALF_ROOT`).

**FHI-aims hooks** — aimshalf does not distribute any FHI-aims code. The installer adds
the aimshalf lines to *your own* (licensed) FHI-aims source tree, then you build as usual:

```bash
python3 fhiaims/install_hooks.py /path/to/fhi-aims.261002 --check   # dry run
python3 fhiaims/install_hooks.py /path/to/fhi-aims.261002            # install
```

Start from a pristine tree (tested: 260331.1 and 261002). `--remove` restores the
original files. `fhiaims/INTEGRATION.md` describes the hooks and how to port them to a
new release. `fhiaims/run_install_tests.sh` checks the build (zero potential ⇒ identical
total energy; the SEP opens the GaAs gap; with `AIMS_STOCK` set, patched = stock
without the keyword).

## Quick start

A calculation folder needs:

- `control.in` — your usual settings, species defaults appended, and `output band ...`
  lines (the gap is read on the band path; the automatic mode also uses them);
- `geometry.in`;
- `aimshalf.yaml` — at least `aims_cmd` (see `examples/aimshalf.yaml`).

```bash
aimshalf execute                  # starts in the background; the terminal can be closed
cat aimshalf_status.txt           # RUNNING / DONE / FAILED, current cut, best so far
tail -f aimshalf_execute.log      # live log
```

Inside a SLURM/PBS/LSF job it runs in the foreground automatically. To resume after a
crash or `kill`, run `aimshalf execute` again: finished runs with matching inputs are
reused.

## Commands

| Command | What it does |
|---|---|
| `execute` | Full correction: choose orbitals, optimize r_cut, report the gap |
| `gen-sep` | Write one SEP file, e.g. `aimshalf gen-sep As -c p=0.5 --cut 3.8` |
| `band-gap` | VBM, CBM and gap from the `band*.out` files (`-i`: fundamental gap) |
| `vbm-character` / `cbm-character` | Mulliken character of the band edge, degenerate states summed (`-i`: edges of the fundamental gap) |
| `band-character K BAND` | Mulliken character of state BAND at k-point K |
| `inspect-mlk` | Show how a `bandmlk*.out` file is read (format check) |

`aimshalf <command> -h` lists the options.

## Choosing the corrections

- **Automatic** (`corrections: auto`, default): a plain-PBE run with Mulliken-resolved bands
  in `pbe_ref/`; the VBM character decides. `correction_code: v` corrects the dominant
  (species, l); `vf` corrects every (species, l) above `valence_threshold` %, sharing the
  1/2 electron by weight (whole percent, as minushalf). Orbitals that are empty in the
  free atom are dropped. An atom with n nearest neighbours of its own species shares the
  hole with them and gets 1/(1+n) of its share (Si: 1/4 e); n comes from `geometry.in`
  and can be forced with `divide_character: [[species, l, n], ...]`, as in minushalf.
- **Explicit**: `corrections: {As: {p: 0.5}}` (valence), optionally with
  `conduction_corrections: {Al: {s: 0.5}}`.

**Conduction corrections**, as in minushalf: a `c` in `correction_code` adds channels
chosen from the CBM character (`c`: dominant orbital; `cf`: every orbital above
`conduction_threshold` %, default 9). The atomic calculation is the same as for the
valence, but the SEP enters with amplitude −|A|: it is repulsive and pushes the
conduction band up. Valence channels are optimized first and kept fixed; the conduction
channels follow, again maximizing the gap. Codes: `v`, `vf`, `c`, `cf`, `vc`, `vfc`,
`vcf`, `vfcf`. `cbm_characters` replaces CBM character values, like `vbm_characters`.

**Which gap**, as in minushalf: `indirect: false` (default) maximizes the smallest
*direct* gap and takes the VBM/CBM character at that k-point; `indirect: true`
maximizes the fundamental gap (VBM and CBM may sit at different k-points). For an
indirect semiconductor (e.g. AlAs) the two give different r_cut. Also as in minushalf:
`overwrite_vbm` / `overwrite_cbm: [k, state]` choose the band edge by hand (k-points
and states numbered as `band-gap` prints them), and `vbm_characters` / `cbm_characters:
[[species, l, percent], ...]` replace character values before the selection.

With several channels, each gets its own atomic calculation and its own r_cut, optimized
one after another (later channels keep the earlier cuts fixed), as in minushalf.

## Output

| File / folder | Content |
|---|---|
| `aimshalf_status.txt` | one-glance state, final result |
| `aimshalf_execute.log` | timestamped log, one block per aims run |
| `aimshalf_results.dat` | r_cut, gap, wall time, state of every run |
| `cut_X.XXX/` | one aims run per r_cut (SEP file, inputs, outputs, fingerprint) |
| `stepN_<species>-<l>[-cond]/` | multi-channel runs, one folder per channel (`-cond`: conduction) |
| `pbe_ref/` | plain-PBE reference (automatic mode); `vbm_character.txt` |
| `stale_*` | runs set aside because their inputs changed (never deleted) |

`scripts/compare_cuts.py` (drop it in the folder and run it) plots gap vs r_cut and the
band structures of all cuts; `scripts/aims_bands_cut.py` plots one calculation.

## Validation

**Validated**
- Conduction correction in FHI-aims 261002 (GaAs, light species, no SOC, `vc`): As-p
  valence 0.54 -> 1.60 eV (r_cut 3.82 bohr), then Ga-s conduction (repulsive SEP)
  -> 2.34 eV (r_cut 3.04 bohr); single, well-defined maxima in both steps.
- SEP vs minushalf (`validation/compare_minushalf.py`): atomic inputs identical for 102
  elements × 4 l; VTOTALs and SEPs identical to ~1e-16 Ha; the q-space term minushalf adds
  to the POTCAR equals the Fourier transform of aimshalf's real-space SEP.
  Multi-channel SEP files equal the sum of minushalf's per-channel SEPs (~5e-16 Ha).
  22/22 checks pass (`validation/results/report.txt`).
- FHI-aims hooks (`fhiaims/run_install_tests.sh`, FHI-aims 261002): a zero potential
  leaves the total energy unchanged to all printed digits; the As-p SEP opens the GaAs
  gap. Comparison with an unpatched binary not yet done.
- Mulliken band-file parser: FHI-aims 260331.1 and 261002, with SOC.

**Not validated / known limitations**
- Atomic potentials are non-relativistic (ATOM), while aims uses atomic ZORA (+ SOC). The
  SEP is a difference of potentials, which cancels much of this, but heavy elements
  (In and beyond) should be checked.
- Mulliken weights per l depend on the basis set (e.g. cation-d weight at the VBM with
  `tight` species): trust the species ranking, not the exact percentages.
- The flat top of gap(r_cut) fixes r_cut only to about ±0.05 bohr; quote it accordingly.
- Total energies and forces with the SEP are not meaningful: use geometries from plain DFT.
- Not tested: GPU builds of aims, spin-polarized systems.
- Differences from minushalf: r_cut found by coarse scan + golden-section search instead
  of Nelder-Mead from a guess; the character of a degenerate band edge is summed over
  the degenerate states; orbitals empty in the free atom are dropped instead of failing.
- Not implemented: shell DFT-1/2, occupation-dependent DFT-1/2.

## Repository layout

```
aimshalf/     Python package (sep, atom_runner, aims_io, workflow, config, cli)
atom/         ATOM solver from minushalf (GPL) + per-element INP inputs
fhiaims/      FHI-aims hook installer, module source, install tests
examples/     annotated aimshalf.yaml
scripts/      drop-in plotting scripts
tests/        pytest suite (+ synthetic aims outputs)
validation/   comparison with minushalf
```

## References and credits

- L. G. Ferreira, M. Marques, L. K. Teles, *Phys. Rev. B* **78**, 125116 (2008).
- L. G. Ferreira, M. Marques, L. K. Teles, *AIP Advances* **1**, 032119 (2011).
- minushalf — GMSN group, ITA: workflow design, the atomic program `atm_cGuima3.f` and the
  atomic input fixtures are taken from it.
- V. Blum *et al.*, *Comput. Phys. Commun.* **180**, 2175 (2009) — FHI-aims.

Author: Eduardo Afonso Moreira Santos (IFSC-USP), advisor Guilherme Matos Sipahi.

License: GPL-3.0 (see `LICENSE`), required by the minushalf components it includes.
