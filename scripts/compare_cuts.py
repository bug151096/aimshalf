#!/usr/bin/env python3
"""
compare_cuts.py  -  compare the band structures of all r_cut runs of an aimshalf execute.

Drop it in the folder that holds the cut_X.XXX/ sub-folders (where you ran
`aimshalf execute`) and run with no arguments:

    python3 compare_cuts.py

Reads every cut_*/ folder that has band files (SOC bands if present, plus the
.no_soc ones). If this folder (or ./pbe_ref, ../pbe_ref from aimshalf auto mode) has band
files of a plain PBE run, they are used as the PBE reference; otherwise the PBE gap is taken from
./aims.out if present (k-grid estimate, marked as such).

Writes, in this folder:
    compare_cuts.dat          table: r_cut, gap (SOC / no-SOC), direct?, VBM/CBM location,
                              CB minima at the labelled points, Delta_SO at Gamma
    compare_gap_vs_cut.png    gap vs r_cut (SOC and no-SOC) + CB minima at G, X, L vs r_cut
    compare_bands_overlay.png all cuts on one plot, coloured by r_cut, optimum in black
    compare_bands_grid.png    one small panel per cut, same scale
    compare_optimum.png       optimal cut vs smallest/largest cut (and PBE bands if available)

Needs numpy and matplotlib.
"""
import os
import re
import glob
import numpy as np

EWIN = (-6, 8)      # energy window (eV) for the plots; edit freely
POINTS = ["G", "X", "L"]   # labelled k-points whose CB minimum is tracked vs r_cut


# ----------------------------------------------------------------------------- reading
def read_lattice(path):
    vecs = [[float(x) for x in l.split()[1:4]] for l in open(path)
            if l.split() and l.split()[0] == "lattice_vector"]
    A = np.array(vecs)
    if A.shape != (3, 3):
        raise SystemExit("Could not read three lattice_vector lines from geometry.in")
    return np.linalg.inv(A).T


def read_segments(path):
    segs = []
    for line in open(path):
        t = line.split()
        if len(t) >= 2 and t[0] == "output" and t[1] == "band":
            labs = t[9:11] if len(t) >= 11 else ["", ""]
            segs.append((labs[0], labs[1] if len(labs) > 1 else ""))
    if not segs:
        raise SystemExit("No 'output band' lines found in control.in")
    return segs


def band_files(calc, nosoc):
    """{segment: file}. nosoc=False -> bandXXXX.out ; True -> bandXXXX.out.no_soc"""
    out = {}
    for fn in glob.glob(os.path.join(calc, "band[0-9][0-9][0-9][0-9].out*")):
        base = os.path.basename(fn)
        m = re.fullmatch(r"band(\d{4})\.out(\.no_soc)?", base)
        if not m or bool(m.group(2)) != nosoc:
            continue
        out[int(m.group(1)) % 100] = fn
    return out


def read_band(fname):
    K, E, O = [], [], []
    for line in open(fname):
        v = line.split()
        if not v:
            continue
        v = [float(x) for x in v]
        lead = 4 if (len(v) - 4) % 2 == 0 else 3
        K.append(v[lead - 3:lead])
        O.append(v[lead::2])
        E.append(v[lead + 1::2])
    return np.array(K), np.array(E), np.array(O)


def load_bands(calc, nosoc=False):
    """Returns dict with x, E (raw eV), O, K, ticks, ticklabels, label_index, segments_x."""
    G = read_lattice(os.path.join(calc, "geometry.in"))
    segs = read_segments(os.path.join(calc, "control.in"))
    files = band_files(calc, nosoc)
    if not files:
        return None
    xs, Es, Os, Ks, ticks, tlab, lab_idx = [], [], [], [], [], [], {}
    xshift, n0 = 0.0, 0
    for i in range(1, len(segs) + 1):
        if i not in files:
            raise SystemExit(f"{calc}: missing band file for segment {i}")
        K, E, O = read_band(files[i])
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(K @ G, axis=0), axis=1))]
        x = d + xshift
        if i == 1:
            ticks.append(x[0]); tlab.append(segs[0][0])
            lab_idx.setdefault(segs[0][0], n0)
        ticks.append(x[-1]); tlab.append(segs[i - 1][1])
        lab_idx.setdefault(segs[i - 1][1], n0 + len(x) - 1)
        xs.append(x); Es.append(E); Os.append(O); Ks.append(K)
        xshift = x[-1]; n0 += len(x)
    return dict(x=np.concatenate(xs), E=np.vstack(Es), O=np.vstack(Os),
                K=np.vstack(Ks), ticks=ticks, ticklabels=tlab, label_index=lab_idx,
                segments_x=xs, segments_E=Es)


def analyse(b):
    """Gap, direct/indirect, where VBM/CBM sit, CB minima and VB maxima at labelled points."""
    E, O, K = b["E"], b["O"], b["K"]
    occ = O > 0.5 * O.max()
    vbm, cbm = E[occ].max(), E[~occ].min()
    ivb = np.unravel_index(np.argmax(np.where(occ, E, -np.inf)), E.shape)[0]
    icb = np.unravel_index(np.argmin(np.where(~occ, E, np.inf)), E.shape)[0]

    def where(i):
        for lab, j in b["label_index"].items():
            if np.allclose(K[j], K[i], atol=1e-3):
                return lab
        return "k=(" + ",".join(f"{v:.3f}" for v in K[i]) + ")"

    points = {}
    for lab, j in b["label_index"].items():
        points[lab] = dict(cb=E[j][~occ[j]].min() - vbm, vb=E[j][occ[j]].max() - vbm)
    # spin-orbit split-off at Gamma (distinct occupied levels below the VBM)
    dso = None
    jg = next((j for l, j in b["label_index"].items() if l.upper() in ("G", "GAMMA")), None)
    if jg is not None:
        lev = np.sort(E[jg][occ[jg]])[::-1]
        distinct = [lev[0]] + [e for k, e in enumerate(lev[1:], 1) if lev[k - 1] - e > 1e-3]
        if len(distinct) > 1:
            dso = distinct[0] - distinct[1]
    return dict(vbm=vbm, cbm=cbm, gap=cbm - vbm,
                direct=np.allclose(K[ivb], K[icb], atol=1e-3),
                vbm_at=where(ivb), cbm_at=where(icb), points=points, split=dso)


def nice(lab):
    return "$\\Gamma$" if lab.upper() in ("G", "GAMMA") else lab



def pbe_gap_from_aimsout(path):
    if not os.path.exists(path):
        return None
    vals = re.findall(r"HOMO-LUMO gap:\s*(-?\d+\.\d+)\s*eV", open(path, errors="replace").read())
    return float(vals[-1]) if vals else None


def plot_bands(ax, b, shift, **kw):
    label = kw.pop("label", None)
    for x, E in zip(b["segments_x"], b["segments_E"]):
        ax.plot(x, E - shift, **kw)
    if label:
        ax.plot([], [], label=label, **kw)      # one legend entry, not one per band


def frame(ax, b, ylabel=True):
    t = b["ticks"]
    for p in t[1:-1]:
        ax.axvline(p, color="#cccccc", lw=0.6, zorder=0)
    ax.axhline(0, color="#888888", lw=0.6, ls="--", zorder=0)
    ax.set_xticks(t, [nice(l) for l in b["ticklabels"]])
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(*EWIN)
    if ylabel:
        ax.set_ylabel("Energy $-$ E$_{VBM}$ (eV)")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    runs = []
    for d in sorted(glob.glob("cut_*"), key=lambda s: float(re.findall(r"\d+\.\d+", s)[0])):
        b = load_bands(d)
        if b is None:
            print(f"skip {d}: no band files")
            continue
        s = load_bands(d, nosoc=True)
        runs.append(dict(cut=float(re.findall(r"\d+\.\d+", d)[0]), dir=d, b=b, a=analyse(b),
                         b0=s, a0=analyse(s) if s else None))
    if not runs:
        raise SystemExit("no cut_*/ folders with band files here")
    pbe_b, pbe_where = None, None
    for cand in (".", "pbe_ref", os.path.join("..", "pbe_ref")):   # aimshalf auto mode writes pbe_ref/
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "band[0-9][0-9][0-9][0-9].out")):
            pbe_b, pbe_where = load_bands(cand), cand
            break
    pbe_a = analyse(pbe_b) if pbe_b else None
    pbe_gap = pbe_a["gap"] if pbe_a else pbe_gap_from_aimsout("aims.out")
    pbe_note = (f"PBE (bands in {pbe_where})" if pbe_a else
                "PBE (aims.out, SCF k-grid estimate)" if pbe_gap is not None else None)
    has_soc = runs[0]["b0"] is not None
    best = max(runs, key=lambda r: r["a"]["gap"])
    cuts = np.array([r["cut"] for r in runs])

    # ------------------------------------------------------------ table
    pts = [p for p in POINTS if p in runs[0]["a"]["points"]]
    with open("compare_cuts.dat", "w") as fh:
        fh.write("# r_cut(bohr)  gap_SOC(eV)  gap_noSOC(eV)  type      VBM_at  CBM_at  "
                 + "  ".join(f"CB_{p}(eV)" for p in pts) + "  Delta_SO_G(eV)\n")
        for r in runs:
            a, a0 = r["a"], r["a0"]
            fh.write(f"{r['cut']:10.3f}  {a['gap']:11.4f}  "
                     f"{(a0['gap'] if a0 else float('nan')):12.4f}  "
                     f"{'direct' if a['direct'] else 'indirect':8s}  {a['vbm_at']:6s}  {a['cbm_at']:6s}  "
                     + "  ".join(f"{a['points'][p]['cb']:9.4f}" for p in pts)
                     + f"  {(a['split'] if a['split'] is not None else float('nan')):13.4f}"
                     + ("   <-- optimum" if r is best else "") + "\n")
        if pbe_gap is not None:
            fh.write(f"# {pbe_note}: gap = {pbe_gap:.4f} eV\n")
    print(open("compare_cuts.dat").read())

    # ------------------------------------------------------------ gap vs cut
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(cuts, [r["a"]["gap"] for r in runs], "o-", color="#1f4e79", lw=2, ms=7,
               label="DFT-1/2, SOC" if has_soc else "DFT-1/2")
    if has_soc:
        ax[0].plot(cuts, [r["a0"]["gap"] for r in runs], "s--", color="#7aa6d1", lw=1.5, ms=6,
                   label="DFT-1/2, no SOC")
    if pbe_gap is not None:
        ax[0].axhline(pbe_gap, color="#888888", ls=":", lw=1.5, label=pbe_note)
    ax[0].plot(best["cut"], best["a"]["gap"], "*", color="black", ms=16, zorder=5,
               label=f"optimum: {best['cut']:.3f} bohr, {best['a']['gap']:.3f} eV")
    ax[0].set_xlabel("r$_{cut}$ (bohr)"); ax[0].set_ylabel("band gap (eV)")
    top = max(max(r["a"]["gap"] for r in runs), max((r["a0"]["gap"] for r in runs if r["a0"]), default=0))
    ax[0].set_ylim(0, top + 0.75)
    ax[0].set_title("Gap vs cut radius"); ax[0].legend(fontsize=8, frameon=False, loc="upper right")
    ax[0].grid(alpha=0.25)
    marks = {"G": "o", "X": "s", "L": "^"}
    shades = ["#0b3a63", "#2f6fa8", "#7aa6d1"]
    for p, c in zip(pts, shades):
        ax[1].plot(cuts, [r["a"]["points"][p]["cb"] for r in runs], marks.get(p, "d") + "-",
                   color=c, lw=1.8, ms=7, label=f"CB minimum at {nice(p)}")
    ax[1].axvline(best["cut"], color="black", lw=0.8, ls=":")
    ax[1].set_xlabel("r$_{cut}$ (bohr)"); ax[1].set_ylabel("E $-$ E$_{VBM}$ (eV)")
    ax[1].set_title("Conduction-band valleys vs cut"); ax[1].legend(fontsize=8, frameon=False)
    ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("compare_gap_vs_cut.png", dpi=200); plt.close(fig)

    # ------------------------------------------------------------ overlay
    norm = colors.Normalize(cuts.min(), cuts.max())
    cmap = colors.LinearSegmentedColormap.from_list("cut", ["#c6dbef", "#08306b"])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for r in runs:
        if r is best:
            continue
        plot_bands(ax, r["b"], r["a"]["vbm"], color=cmap(norm(r["cut"])), lw=1.0)
    plot_bands(ax, best["b"], best["a"]["vbm"], color="black", lw=2.0,
               label=f"optimum r$_{{cut}}$ = {best['cut']:.3f} bohr")
    if pbe_a:
        plot_bands(ax, pbe_b, pbe_a["vbm"], color="#b2182b", lw=1.2, ls="--", label="PBE")
    frame(ax, best["b"])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, pad=0.02).set_label("r$_{cut}$ (bohr)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("All cuts" + (" (SOC bands)" if has_soc else ""), fontsize=11)
    fig.tight_layout(); fig.savefig("compare_bands_overlay.png", dpi=200); plt.close(fig)

    # ------------------------------------------------------------ grid
    n = len(runs) + (1 if pbe_a else 0)
    ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.6 * nrow), squeeze=False)
    panels = ([("PBE", pbe_b, pbe_a, "#b2182b")] if pbe_a else []) + \
             [(f"r$_{{cut}}$ = {r['cut']:.3f}", r["b"], r["a"],
               "black" if r is best else "#1f4e79") for r in runs]
    for k, (name, b, a, c) in enumerate(panels):
        axk = axs[k // ncol][k % ncol]
        plot_bands(axk, b, a["vbm"], color=c, lw=1.0)
        frame(axk, b, ylabel=(k % ncol == 0))
        axk.set_title(f"{name}\n$E_g$ = {a['gap']:.3f} eV" + ("  (optimum)" if c == "black" else ""),
                      fontsize=9)
        axk.tick_params(labelsize=8)
    for k in range(len(panels), nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    fig.tight_layout(); fig.savefig("compare_bands_grid.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------ optimum vs extremes
    fig, ax = plt.subplots(figsize=(6, 6.5))
    lo, hi = runs[0], runs[-1]
    plot_bands(ax, lo["b"], lo["a"]["vbm"], color="#9ecae1", lw=1.0,
               label=f"r$_{{cut}}$ = {lo['cut']:.3f} ($E_g$ {lo['a']['gap']:.3f} eV)")
    plot_bands(ax, hi["b"], hi["a"]["vbm"], color="#4a7fb5", lw=1.0, ls="-.",
               label=f"r$_{{cut}}$ = {hi['cut']:.3f} ($E_g$ {hi['a']['gap']:.3f} eV)")
    if pbe_a:
        plot_bands(ax, pbe_b, pbe_a["vbm"], color="#b2182b", lw=1.2, ls="--",
                   label=f"PBE ($E_g$ {pbe_a['gap']:.3f} eV)")
    plot_bands(ax, best["b"], best["a"]["vbm"], color="black", lw=2.0,
               label=f"optimum {best['cut']:.3f} ($E_g$ {best['a']['gap']:.3f} eV)")
    frame(ax, best["b"])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig("compare_optimum.png", dpi=200); plt.close(fig)

    print("wrote compare_cuts.dat, compare_gap_vs_cut.png, compare_bands_overlay.png, "
          "compare_bands_grid.png, compare_optimum.png")
