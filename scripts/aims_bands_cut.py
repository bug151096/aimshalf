#!/usr/bin/env python3
"""
aims_bands_cut.py  -  band structure of ONE FHI-aims calculation (e.g. one cut_X.XXX folder).

Drop it in the calculation folder (or run it from there) and run with no arguments:

    python3 aims_bands_cut.py

It reads the k-path and labels from control.in, the lattice from geometry.in,
and the band files. With spin-orbit coupling it uses the SOC bands
(bandXXXX.out) and also reports the scalar ones (bandXXXX.out.no_soc).
Energies are shifted so the VBM is at 0 eV.

Writes, in the same folder:
    bands.dat          k-distance + one column per band (SOC if present), VBM = 0
    bands_nosoc.dat    same for the scalar-relativistic bands (only with SOC runs)
    bands.gnu          ready-to-edit gnuplot script   (gnuplot bands.gnu -> bands.png)
    bands_preview.png  matplotlib plot (SOC solid, scalar dashed)
    gap.txt            gap, direct/indirect, VBM/CBM locations, CB minima at the labelled points

Needs numpy; matplotlib for the preview.
"""
import os
import re
import glob
import numpy as np

CALC = "."
EWIN = (-6, 8)      # energy window (eV) for the plots; edit freely


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


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    soc = load_bands(CALC, nosoc=False)
    if soc is None:
        raise SystemExit("No band files (bandXXXX.out) in this folder")
    scal = load_bands(CALC, nosoc=True)          # None for runs without SOC
    has_soc = scal is not None

    m = re.search(r"cut_(\d+\.\d+)", os.path.abspath(CALC))
    seps = glob.glob(os.path.join(CALC, "sep_*.dat"))
    title = (f"r_cut = {m.group(1)} bohr" if m else os.path.basename(os.path.abspath(CALC)))

    a = analyse(soc)
    lines = [f"folder        : {os.path.abspath(CALC)}",
             f"bands used    : {'SOC (bandXXXX.out)' if has_soc else 'scalar (bandXXXX.out)'}",
             f"SEP files     : {', '.join(os.path.basename(s) for s in seps) or 'none (plain DFT)'}",
             f"gap           : {a['gap']:.4f} eV  ({'direct' if a['direct'] else 'indirect'}, "
             f"VBM at {a['vbm_at']}, CBM at {a['cbm_at']})"]
    if has_soc:
        a0 = analyse(scal)
        lines.append(f"gap no-SOC    : {a0['gap']:.4f} eV  ({'direct' if a0['direct'] else 'indirect'})")
        if a["split"] is not None:
            lines.append(f"Delta_SO (G)  : {a['split']:.4f} eV  (spin-orbit split-off at Gamma)")
    lines.append("CB minimum / VB maximum at labelled points (eV, VBM = 0):")
    for lab, p in a["points"].items():
        lines.append(f"    {lab:6s} CB {p['cb']:8.4f}   VB {p['vb']:8.4f}")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(CALC, "gap.txt"), "w").write(text + "\n")

    nb = soc["E"].shape[1]
    hdr = "k_dist " + " ".join(f"b{j + 1}" for j in range(nb))
    np.savetxt(os.path.join(CALC, "bands.dat"),
               np.column_stack([soc["x"], soc["E"] - a["vbm"]]), header=hdr, fmt="%.6f")
    if has_soc:
        hdr0 = "k_dist " + " ".join(f"b{j + 1}" for j in range(scal["E"].shape[1]))
        np.savetxt(os.path.join(CALC, "bands_nosoc.dat"),
                   np.column_stack([scal["x"], scal["E"] - analyse(scal)["vbm"]]),
                   header=hdr0, fmt="%.6f")

    t, tl = soc["ticks"], soc["ticklabels"]
    gl = lambda s: "{/Symbol G}" if s.upper() in ("G", "GAMMA") else s
    xt = ", ".join(f'"{gl(l)}" {p:.6f}' for p, l in zip(t, tl))
    arrows = "\n".join(f"set arrow from {p:.6f},{EWIN[0]} to {p:.6f},{EWIN[1]} nohead "
                       f"lc rgb '#888888' lw 1" for p in t[1:-1])
    gnu = f"""# bands.gnu  -  edit freely, then run:  gnuplot bands.gnu
set terminal pngcairo size 600,800 enhanced font 'Helvetica,16'
set output 'bands.png'
# vector version:  set terminal pdfcairo size 4,5 enhanced font 'Helvetica,14' ; set output 'bands.pdf'
set title "{title}   E_g = {a['gap']:.3f} eV"
set xrange [{t[0]:.6f}:{t[-1]:.6f}]
set yrange [{EWIN[0]}:{EWIN[1]}]
set ylabel "Energy - E_{{VBM}}  (eV)"
set xtics ({xt})
unset key
set arrow from {t[0]:.6f},0 to {t[-1]:.6f},0 nohead lc rgb '#cc0000' lw 1 dt 2
{arrows}
nbands = {nb}
plot for [i=2:nbands+1] 'bands.dat' using 1:i with lines lw 2 lc rgb '#1f4e79'
"""
    open(os.path.join(CALC, "bands.gnu"), "w").write(gnu)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 6))
        if has_soc:
            v0 = analyse(scal)["vbm"]
            for x, E in zip(scal["segments_x"], scal["segments_E"]):
                ax.plot(x, E - v0, color="#999999", lw=0.9, ls="--")
        for x, E in zip(soc["segments_x"], soc["segments_E"]):
            ax.plot(x, E - a["vbm"], color="#1f4e79", lw=1.3)
        for p in t:
            ax.axvline(p, color="k", lw=0.5)
        ax.axhline(0, color="r", lw=0.7, ls="--")
        ax.set_xticks(t, [nice(l) for l in tl])
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(*EWIN)
        ax.set_ylabel("Energy $-$ E$_{VBM}$ (eV)")
        ax.set_title(f"{title}   $E_g$ = {a['gap']:.3f} eV" +
                     ("\n(solid: SOC, dashed: no SOC)" if has_soc else ""), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(CALC, "bands_preview.png"), dpi=200)
        print("wrote bands.dat, bands.gnu, bands_preview.png, gap.txt")
    except ImportError:
        print("wrote bands.dat, bands.gnu, gap.txt  (matplotlib not found: no preview)")
