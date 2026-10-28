# Changelog

## 0.3.0 — 2026-10-30

Parity with minushalf: band edges, shared holes, conduction corrections.

### Changed
- The control.in keyword of the FHI-aims hook is now `aimshalf_potential` (the old
  name `external_atomic_potential` is still accepted). **Rebuild FHI-aims** with the
  current hooks; `execute` stops with a clear message if the aims binary does not know
  the new keyword.
- FHI-aims is no longer patched with `.patch` files, which contained lines of FHI-aims
  source: `fhiaims/install_hooks.py` adds the aimshalf lines to the user's own FHI-aims
  tree (`--check`, `--remove`). On pristine 260331.1 and 261002 the result is
  byte-identical to the former patches. aimshalf now distributes no FHI-aims code.
- `indirect: false` is now the default, as in minushalf: the smallest **direct** gap is
  maximized and the VBM/CBM character is taken at that k-point. Set `indirect: true` for
  the 0.2 behaviour (fundamental gap).
- `vf` splits the half electron in whole percent, `round(100 p / sum p)`, as minushalf.
- `band-gap` prints k-point and state numbers (used by `band-character` and `overwrite_*`).

### Added
- Conduction corrections, as in minushalf: correction codes `c`, `cf`, `vc`, `vfc`,
  `vcf`, `vfcf`. Conduction channels come from the CBM character (`cf`: above
  `conduction_threshold`, default 9 %), use amplitude -|A| (repulsive) and are optimized
  after the valence channels, which stay fixed. `cbm_characters`; explicit mode with
  `conduction_corrections: {Al: {s: 0.5}}`.
- `divide_character`: an atom with n nearest neighbours of its own species gets
  1/(1+n) of its share of the half electron (Si: 1/4 e); n from geometry.in or forced
  with `divide_character: [[species, l, n], ...]`, as in minushalf.
- `indirect` option; `-i` in `band-gap`, `vbm-character`, `cbm-character`.
- `band-character K BAND` command.
- `overwrite_vbm`, `overwrite_cbm`, `vbm_characters` options.


## 0.2.0 — 2026-10-28

### Fixed
- Finished runs (`cut_X.XXX/`, `pbe_ref/`) are reused only if they were made from the
  same inputs. Each run folder stores `aimshalf_fingerprint.json` (hashes of control.in,
  geometry.in, SEP files, and the aims binary). A folder that does not match is renamed
  `stale_<name>_<time>/` (never deleted) and the run is repeated. Before, any finished
  `aims.out` was reused, even after changing the aims version or the correction.
- `vbm-character` / `cbm-character` sum the character over degenerate states (e.g. the
  4-fold Γ8 VBM with SOC); a single state of a degenerate set has an arbitrary composition.
- `aimshalf.yaml` is read from the `--dir` folder, not only from the current folder.
- `pyyaml` declared as a dependency.
- A non-editable install (`pip install .`) now fails with a clear message; the source
  folder can also be given with `AIMSHALF_ROOT`.

### Changed
- `gen-sep` with several `-c` options builds one SEP per channel (own atomic run) and sums
  them, the same scheme `execute` uses (and minushalf). Before, all charge was removed in
  a single atomic calculation.
- `vbm-character` / `cbm-character` read all `bandmlk*.out` of `--dir` by default;
  `--k-index` removed (the edge is located automatically).
- `minushalf.yaml` is no longer read (its keys are not aimshalf's).
- Help text for every command; `aimshalf --version`.
- Validation L2 is now a pass/fail check: aimshalf's multi-channel SEP file vs the sum
  of minushalf's per-channel SEPs (was an informational comparison of the old scheme).
  22/22 checks pass.

### Added
- Pre-flight checks in `execute` (missing inputs, aims binary, atomic solver, band
  lines, species and orbitals, xc mismatch) before anything starts.
- Warning when the optimal r_cut is at the edge of the search range.
- `reuse_unverified` option (reuse pre-0.2 runs without fingerprint).
- Automated tests (`pytest`): run reuse/invalidation, multi-channel SEP, parsers.
- Optional dependencies: `pip install -e ".[dev]"` (pytest, matplotlib).
- `examples/aimshalf.yaml`, rewritten README, CHANGELOG, GPL-3.0 LICENSE.

### Repository
- FHI-aims 261002 patch (`fhiaims/aimshalf-fhiaims-261002.patch`); the 260331.1 patch
  does not apply to 261002. The module is now built only in library `aims4` (it was
  listed twice in CMakeLists.txt). Install tests pass on 261002.
- `fortran/` and `cluster_install/` merged into `fhiaims/`; the reference copy of the
  module now matches the one in the patches.
- `atom/fixtures/` keeps only the per-element `INP` files aimshalf uses (minushalf's VASP
  test data, POTCARs and VTOTALs removed); `demo/` removed.

## 0.1.0 — 2026-10-02
First working version: SEP generation with minushalf's atomic solver, FHI-aims patch
(`external_atomic_potential`), `execute` with coarse scan + golden-section r_cut
optimization, automatic channel selection (`v`, `vf`) from a PBE Mulliken run, one r_cut
per channel, status/log/results files, background execution and restart.
