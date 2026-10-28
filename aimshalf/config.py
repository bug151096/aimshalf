"""
aimshalf/config.py  --  YAML config loader for `aimshalf execute`.

Config search order (first found wins):
  1. --config <path>  (explicit)
  2. aimshalf.yaml in the --dir folder (default: the current folder)
Any value given on the command line overrides the same value from the file.
"""
import os

try:
    import yaml
except ImportError:
    yaml = None

DEFAULTS = {
    "dir": ".",
    "corrections": None,
    "aims_cmd": "mpirun -np 4 aims.x",
    "xc": "pb",
    "cut_min": 1.5,
    "cut_max": 5.0,
    "tol": 0.05,
    "coarse_points": 5,
    "use_band_files": True,
    "power": 8,
    "amplitude": 1.0,
    "gap_tol": 0.005,        # eV: stop refining when the top of gap(r_cut) is this flat
    "cut_resolution": 0.01,  # bohr: cuts are rounded to this; never re-run the same cut
    "max_iter": 12,          # golden-section steps after the coarse scan
    "finish_grace": 30,      # s: kill a launcher still alive this long after aims finished
    "run_timeout": 0,        # s per aims run, 0 = no limit
    # automatic correction selection (used when 'corrections' is empty or "auto")
    "correction_code": "v",      # minushalf codes: v, vf, c, cf, vc, vfc, vcf, vfcf
                                 # v/c: dominant VBM/CBM orbital; f: all above threshold
    "valence_threshold": 10.0,   # % (minushalf fractional_valence_treshold)
    "conduction_threshold": 9.0, # % (minushalf fractional_conduction_treshold)
    "conduction_corrections": None,  # explicit mode: {Ga: {s: 0.5}} (needs `corrections`)
    "pbe_ref_dir": "pbe_ref",    # folder of the plain-DFT Mulliken reference run
    "reuse_unverified": False,   # reuse finished runs that have no fingerprint (pre-0.2)
    "indirect": False,           # False: smallest direct gap (minushalf default);
                                 # True: fundamental gap (VBM and CBM may differ in k)
    "overwrite_vbm": None,       # [k, state]: use this band as the VBM (minushalf)
    "overwrite_cbm": None,       # [k, state]: use this band as the CBM (minushalf)
    "vbm_characters": None,      # [[species, l, percent], ...]: replace VBM characters
    "cbm_characters": None,      # [[species, l, percent], ...]: replace CBM characters
    "divide_character": None,    # [[species, l, n], ...]: force the number of equal
                                 # nearest neighbours (default: from geometry.in)
    "cut_ranges": None,          # optional per species, e.g. {N: [1.5, 4.0], Al: [2.5, 5.0]}
}

CONFIG_NAMES = ("aimshalf.yaml",)


def find_config(explicit=None, start_dir="."):
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"config file not found: {explicit}")
        return explicit
    for name in CONFIG_NAMES:
        cand = os.path.join(start_dir, name)
        if os.path.isfile(cand):
            return cand
    return None


def load_config(explicit=None, start_dir="."):
    """Return (settings_dict, path) merged over DEFAULTS. Missing file -> defaults."""
    cfg = dict(DEFAULTS)
    path = find_config(explicit, start_dir)
    if path is None:
        return cfg, None
    if yaml is None:
        raise ImportError("pyyaml is required to read config files: "
                          "pip install pyyaml")
    with open(path) as fh:
        loaded = yaml.safe_load(fh) or {}
    if "execute" in loaded and isinstance(loaded["execute"], dict):
        loaded = {**{k: v for k, v in loaded.items() if k != "execute"},
                  **loaded["execute"]}
    unknown = set(loaded) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    cfg.update(loaded)
    return cfg, path
