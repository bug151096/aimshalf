"""
Parsers for FHI-aims inputs/outputs needed by the DFT-1/2 workflow.

VERIFY-AGAINST-YOUR-VERSION notes
---------------------------------
Formats below were written against publicly documented FHI-aims conventions;
aims output evolves between releases. Each parser is deliberately tolerant
and raises with a helpful message when a pattern is not found. The Mulliken
band-projection format in particular (bandmlk*.out, produced together with
`output band ...` when Mulliken decomposition is requested) MUST be validated
against one real file from your aims build before trusting the character
analysis; see MullikenBandFile docstring.
"""
import glob
import os
import re
import numpy as np

HARTREE_EV = 27.211384500

# --------------------------------------------------------------------------
# geometry.in
# --------------------------------------------------------------------------

def parse_geometry(path="geometry.in"):
    """Return list of dicts [{'index':1,'species':'Ga','xyz':(...)}, ...] (1-based)."""
    atoms = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            tok = line.split()
            if tok[0] in ("atom", "atom_frac"):
                atoms.append({
                    "index": len(atoms) + 1,
                    "species": tok[4],
                    "xyz": tuple(float(x) for x in tok[1:4]),
                    "frac": tok[0] == "atom_frac",
                })
    if not atoms:
        raise ValueError(f"No atoms found in {path}")
    return atoms


def lattice_vectors(path="geometry.in"):
    """3x3 array of lattice vectors (rows, Angstrom), or None for a cluster."""
    vecs = []
    with open(path) as fh:
        for line in fh:
            tok = line.split("#")[0].split()
            if tok and tok[0] == "lattice_vector":
                vecs.append([float(x) for x in tok[1:4]])
    return np.array(vecs) if len(vecs) == 3 else None


def equal_neighbors(path, species, shell_tol=1.15):
    """
    Number of distinct atoms of the same species among the nearest neighbours of
    the first `species` atom in geometry.in (periodic images included) -- the
    quantity minushalf takes from the OUTCAR neighbour table for divide_character.
    Nearest neighbours: distance < shell_tol x the nearest-neighbour distance.
    0 for III-V compounds; 1 for Si/diamond in the 2-atom cell (hole shared: 1/4 e).
    """
    atoms = parse_geometry(path)
    lat = lattice_vectors(path)
    if any(a["frac"] for a in atoms) and lat is None:
        raise ValueError(f"atom_frac without three lattice_vector lines in {path}")
    pos = np.array([np.array(a["xyz"]) @ lat if a["frac"] else np.array(a["xyz"])
                    for a in atoms])
    first = next((i for i, a in enumerate(atoms) if a["species"] == species), None)
    if first is None:
        raise ValueError(f"species {species} not in {path}")
    shifts = [np.zeros(3)] if lat is None else [
        np.array([i, j, k]) @ lat for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    dist = []
    for j in range(len(atoms)):
        for T in shifts:
            d = float(np.linalg.norm(pos[j] + T - pos[first]))
            if d > 1e-6:
                dist.append((j, d))
    dmin = min(d for _, d in dist)
    shell = {j for j, d in dist if d < shell_tol * dmin}
    return sum(1 for j in shell if j != first and atoms[j]["species"] == species)


def atoms_map(path="geometry.in"):
    """{atom_index(1-based): species_symbol} - minushalf's get_atoms_map analog."""
    return {a["index"]: a["species"] for a in parse_geometry(path)}


# --------------------------------------------------------------------------
# aims.out (standard output)
# --------------------------------------------------------------------------

_PATTERNS = {
    "fermi": [
        r"\|\s*Chemical potential \(Fermi level\)\s*:?\s*(-?\d+\.\d+)\s*eV",
        r"\|\s*Chemical potential\s*:?\s*(-?\d+\.\d+)\s*eV",
    ],
    "vbm": [r"Highest occupied state \(VBM\)(?: at)?\s*:?\s*(-?\d+\.\d+)\s*eV"],
    "cbm": [r"Lowest unoccupied state \(CBM\)(?: at)?\s*:?\s*(-?\d+\.\d+)\s*eV"],
    "gap": [
        r"(?:Estimated overall HOMO-LUMO|Overall HOMO-LUMO|HOMO-LUMO) gap\s*:?\s*(-?\d+\.\d+)\s*eV",
        r"ESTIMATED overall HOMO-LUMO gap:\s*(-?\d+\.\d+)\s*eV",
    ],
    "converged": [r"Self-consistency cycle converged"],
}


class AimsStdout:
    """Parse the quantities we need from an aims standard-output file."""

    def __init__(self, path="aims.out"):
        self.path = path
        with open(path, errors="replace") as fh:
            self.text = fh.read()

    def _last(self, key, required=True):
        val = None
        for pat in _PATTERNS[key]:
            found = re.findall(pat, self.text)
            if found:
                val = found[-1]
        if val is None and required:
            raise ValueError(
                f"Pattern for '{key}' not found in {self.path}. "
                f"Check the aims version's exact output wording and extend "
                f"_PATTERNS in aims_io.py.")
        return val

    @property
    def converged(self):
        return bool(re.search(_PATTERNS["converged"][0], self.text))

    @property
    def fermi_ev(self):
        return float(self._last("fermi"))

    @property
    def gap_ev(self):
        """Scalar gap printed by aims (from the SCF k-grid).
        For band-path gaps prefer BandStructure below."""
        return float(self._last("gap"))

    @property
    def vbm_ev(self):
        return float(self._last("vbm"))

    @property
    def cbm_ev(self):
        return float(self._last("cbm"))


# --------------------------------------------------------------------------
# band####.out  (from `output band ...`)
# Line format: idx  k1 k2 k3  occ_1 e_1(eV)  occ_2 e_2(eV) ...
# Spin-polarized: band1001/band2001 pairs (spin up/down).
# --------------------------------------------------------------------------

class BandStructure:
    def __init__(self, paths):
        """paths: list of band*.out files (all segments, one spin channel
        per call; pass both channels' files together for a global gap)."""
        self.kpoints = []   # (kx,ky,kz)
        self.occ = []       # per k: np.array
        self.eig = []       # per k: np.array (eV)
        for p in sorted(paths):
            with open(p) as fh:
                for line in fh:
                    tok = line.split()
                    if len(tok) < 6:
                        continue
                    vals = np.array(tok[1:], float)
                    self.kpoints.append(tuple(vals[:3]))
                    pairs = vals[3:].reshape(-1, 2)
                    self.occ.append(pairs[:, 0])
                    self.eig.append(pairs[:, 1])
        if not self.eig:
            raise ValueError("No band data parsed; check band*.out paths")

    @classmethod
    def from_dir(cls, directory=".", spin=None):
        pat = "band*.out" if spin is None else f"band{spin}*.out"
        files = [f for f in glob.glob(os.path.join(directory, pat))
                 if "mlk" not in os.path.basename(f)]
        return cls(files)

    def band_edges(self, occ_threshold=0.5, indirect=True):
        """
        VBM and CBM (eV) with their k-point index and band index (both 0-based).

        indirect=True : fundamental gap; VBM and CBM may sit at different k-points.
        indirect=False: smallest direct gap; VBM and CBM at the same k-point
                        (minushalf's default, `indirect: false`).
        """
        per_k = []    # (k, vbm band, vbm energy, cbm band, cbm energy)
        for ik, (occ, eig) in enumerate(zip(self.occ, self.eig)):
            filled = np.where(occ > occ_threshold)[0]
            empty = np.where(occ <= occ_threshold)[0]
            if filled.size and empty.size:
                iv = filled[np.argmax(eig[filled])]
                ic = empty[np.argmin(eig[empty])]
                per_k.append((ik, int(iv), float(eig[iv]), int(ic), float(eig[ic])))
        if not per_k:
            raise ValueError("no k-point with both occupied and empty states")
        if indirect:
            kv = max(per_k, key=lambda t: t[2])
            kc = min(per_k, key=lambda t: t[4])
        else:
            kv = kc = min(per_k, key=lambda t: t[4] - t[2])
        out = {"vbm": kv[2], "vbm_k": kv[0], "vbm_band": kv[1],
               "cbm": kc[4], "cbm_k": kc[0], "cbm_band": kc[3]}
        out["gap"] = out["cbm"] - out["vbm"]
        out["direct"] = out["vbm_k"] == out["cbm_k"]
        return out


# --------------------------------------------------------------------------
# Mulliken-decomposed band files (bandmlk####.out)
# --------------------------------------------------------------------------

class MullikenBandFile:
    """
    Parser for FHI-aims Mulliken-decomposed band files (bandmlk####.out,
    written by `output band_mulliken ...`).

    Validated against FHI-aims 260331 reference files (SOC). Layout per k-point:

        k point number:     1: (   0.1   0.11   0.12 )
            State       eigenvalue  occ.number atom       spin       total
            State        1
                1   -11876.09   1.0000000   1   1   0.00   0.00   0.00   0.00
                ...  (one line per atom, and per spinor component with SOC)

    The column layout is read from the header line: with SOC (or spin
    polarization) a `spin` column sits between `atom` and `total`; without it,
    `total` follows `atom` directly. Columns after `total` are l = s, p, d, f, g...
    """
    L_LABELS = ("s", "p", "d", "f", "g", "h")

    def __init__(self, path):
        self.path = path
        self.kpoints = []      # [(k1,k2,k3)]
        self.blocks = []       # per k: {state: {"eig":e, "occ":o, "atoms":{atom:{l:w,"total":w}}}}
        self.has_spin_column = None
        self._parse()

    def _parse(self):
        current, total_col = None, None
        kre = re.compile(r"k point number:\s*\d+:\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)")
        with open(self.path) as fh:
            for line in fh:
                m = kre.search(line)
                if m:
                    current = {}
                    self.blocks.append(current)
                    self.kpoints.append(tuple(float(x) for x in m.groups()))
                    continue
                if "eigenvalue" in line and "atom" in line:
                    cols = line.split()
                    total_col = cols.index("total")
                    self.has_spin_column = "spin" in cols
                    continue
                tok = line.split()
                if current is None or total_col is None or not tok or tok[0] == "State":
                    continue
                try:
                    state = int(tok[0]); eig = float(tok[1]); occ = float(tok[2])
                    atom = int(tok[3])
                    vals = [float(x) for x in tok[total_col:]]
                except (ValueError, IndexError):
                    continue
                st = current.setdefault(state, {"eig": eig, "occ": occ, "atoms": {}})
                entry = st["atoms"].setdefault(atom, {"total": 0.0})
                entry["total"] += vals[0]          # summed over spinor components
                for lbl, w in zip(self.L_LABELS, vals[1:]):
                    entry[lbl] = entry.get(lbl, 0.0) + w
        if not self.blocks:
            raise ValueError(f"no k-point blocks found in {self.path}")

    def get_band_projection(self, kpoint_index, band_number):
        """{atom_index: {'s': w, 'p': w, ...}} for a k index (0-based) and state (1-based)."""
        st = self.blocks[kpoint_index].get(band_number)
        return {a: {l: w for l, w in d.items() if l != "total"}
                for a, d in st["atoms"].items()} if st else {}


def edge_character(mlk_files, which="vbm", degeneracy_tol=1e-3, indirect=True,
                   location=None):
    """
    Orbital character of a band edge from bandmlk files alone.

    k-points are numbered 1, 2, ... over all files in order (the same numbering
    `band-gap` prints). An edge is the highest filled state (VBM; occupation above
    half the maximum) or the lowest empty one (CBM):
      indirect=True  : over all k-points (fundamental gap)
      indirect=False : at the k-point of the smallest direct gap (minushalf default)
      location=(k, state): that state at that k-point, whatever `which` is
                           (minushalf overwrite_vbm / overwrite_cbm)
    The character is summed over all states degenerate with the edge at that
    k-point (within `degeneracy_tol` eV): with SOC the GaAs-like VBM at Gamma is
    4-fold, and a single state of a degenerate set has an arbitrary composition.
    Returns (info, {atom: {l: weight}}).
    """
    files = [MullikenBandFile(f) if isinstance(f, str) else f for f in mlk_files]
    blocks = [(fi, ik, blk) for fi, mf in enumerate(files) for ik, blk in enumerate(mf.blocks)]
    occ_max = max(st["occ"] for _, _, blk in blocks for st in blk.values())

    if location is not None:
        g, s0 = int(location[0]), int(location[1])
        if not 1 <= g <= len(blocks):
            raise ValueError(f"k-point {g} out of range 1..{len(blocks)}")
        fi, ik, blk = blocks[g - 1]
        if s0 not in blk:
            raise ValueError(f"state {s0} not found at k-point {g}")
        e0 = blk[s0]["eig"]
    else:
        cand = []    # (k number, (E_vbm, state), (E_cbm, state))
        for g, (fi, ik, blk) in enumerate(blocks, 1):
            filled = [(st["eig"], s) for s, st in blk.items() if st["occ"] > 0.5 * occ_max]
            empty = [(st["eig"], s) for s, st in blk.items() if st["occ"] <= 0.5 * occ_max]
            if filled and empty:
                cand.append((g, max(filled), min(empty)))
        if not cand:
            raise ValueError(f"could not locate the {which.upper()} in the bandmlk files")
        if not indirect:
            g, v, c = min(cand, key=lambda t: t[2][0] - t[1][0])
        elif which == "vbm":
            g, v, c = max(cand, key=lambda t: t[1][0])
        else:
            g, v, c = min(cand, key=lambda t: t[2][0])
        e0, s0 = v if which == "vbm" else c
        fi, ik, blk = blocks[g - 1]

    states = sorted(s for s, st in blk.items() if abs(st["eig"] - e0) < degeneracy_tol)
    proj = {}
    for s in states:
        for atom, d in blk[s]["atoms"].items():
            tgt = proj.setdefault(atom, {})
            for l, w in d.items():
                if l != "total":
                    tgt[l] = tgt.get(l, 0.0) + w
    info = dict(energy=e0, file=files[fi].path, k_index=ik, k_number=g,
                kpoint=files[fi].kpoints[ik], states=states)
    return info, proj


def character_table(projection, atom_species):
    """
    Percent contribution per (atom, l) - the analog of minushalf's vbm-character
    table. projection: {atom_idx: {l: w}}; atom_species: {atom_idx: symbol}.
    Negative Mulliken weights (possible) are clipped to zero before normalizing.
    """
    rows, total = [], 0.0
    for atom, ws in projection.items():
        for l, w in ws.items():
            if l != "total":
                total += max(w, 0.0)
    for atom, ws in projection.items():
        for l, w in ws.items():
            if l == "total" or w <= 0:
                continue
            rows.append({"atom": atom, "species": atom_species.get(atom, "?"),
                         "l": l, "weight": w,
                         "percent": 100.0 * w / total if total > 0 else 0.0})
    rows.sort(key=lambda r: -r["percent"])
    return rows


def species_character(rows):
    """Collapse a character table to {(species, l): percent}, summed over atoms."""
    out = {}
    for r in rows:
        key = (r["species"], r["l"])
        out[key] = out.get(key, 0.0) + r["percent"]
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
