"""
Self-energy potential (SEP) construction for DFT-1/2 in FHI-aims.

Conventions
-----------
- The ATOM program (atm_cGuima3, L. G. Ferreira's modified ATOM) writes
  VTOTAL files containing r*V(r) in Rydberg*bohr on a logarithmic radial
  grid (r*V -> -2Z as r -> 0).
- SEP definition (minushalf/Ferreira convention, "valence" correction):
      V_S(r) = [V_ion(r) - V_atom(r)] * Theta(r)
  where V_ion is the KS potential of the atom with fractional occupation
  (e.g. -1/2 e removed from the orbital dominating the VBM) and V_atom the
  neutral ground-state atom. This is attractive (~ -q/r Ha tail before
  trimming, q = removed charge) and, added at the hole-hosting (anion)
  sites, lowers the VBM and opens the gap.
- Trimming: Theta(r) = A*[1 - (r/r_cut)^p]^3 for r <= r_cut, 0 beyond
  (p = 8 default). Conduction (+1/2) correction = amplitude -1.
- Output for FHI-aims: r in bohr, V_S in Hartree (Ry/2).
"""
import re
import numpy as np

RY_TO_HA = 0.5
TRIM_EXPONENT_DEFAULT = 8


class Vtotal:
    """Radial grid + r*V(r) [Ry*bohr] parsed from an ATOM VTOTAL file."""

    def __init__(self, radius: np.ndarray, rv_down: np.ndarray):
        self.radius = np.asarray(radius, dtype=float)
        self.rv_down = np.asarray(rv_down, dtype=float)
        if self.radius.shape != self.rv_down.shape:
            raise ValueError("radius/potential length mismatch in VTOTAL")

    @classmethod
    def from_file(cls, filename: str) -> "Vtotal":
        radius, down = [], []
        section = None
        with open(filename) as fh:
            lines = iter(fh)
            for line in lines:
                if re.match(r"^\s*Raios", line):
                    section = "radius"
                    continue
                if re.search(r"Down\s+potential\s+follows", line):
                    section = "down"
                    next(lines)  # skip the l-value line
                    continue
                if re.search(r"Up\s+potential\s+follows", line):
                    break
                if section == "radius":
                    try:
                        radius.extend(float(x) for x in line.split())
                    except ValueError:
                        section = None
                elif section == "down":
                    try:
                        down.extend(float(x) for x in line.split())
                    except ValueError:
                        pass
        if not radius or not down:
            raise ValueError(f"Could not parse VTOTAL file: {filename}")
        n = min(len(radius), len(down))
        return cls(np.array(radius[:n], float), np.array(down[:n], float))

    def potential_hartree(self) -> np.ndarray:
        """V(r) in Hartree (divides out r, converts Ry -> Ha)."""
        return RY_TO_HA * self.rv_down / self.radius


def trimming_theta(radius, r_cut, power=TRIM_EXPONENT_DEFAULT, amplitude=1.0):
    """Theta(r) = A*[1-(r/r_cut)^p]^3, 0 for r >= r_cut."""
    r = np.asarray(radius, float)
    theta = amplitude * np.power(1.0 - np.power(r / r_cut, power), 3)
    theta[r >= r_cut] = 0.0
    return theta


class SelfEnergyPotential:
    """Trimmed SEP on the atomic radial grid, in Hartree vs bohr."""

    def __init__(self, vtotal_atom: Vtotal, vtotal_ion: Vtotal):
        if len(vtotal_atom.radius) != len(vtotal_ion.radius) or not np.allclose(
                vtotal_atom.radius, vtotal_ion.radius, rtol=1e-8):
            # Grids should coincide (same element, same code); interpolate if not.
            r = vtotal_atom.radius
            rv_ion = np.interp(r, vtotal_ion.radius, vtotal_ion.rv_down)
            vtotal_ion = Vtotal(r, rv_ion)
        self.radius = vtotal_atom.radius
        # (V_ion - V_atom) in Hartree
        self.raw = RY_TO_HA * (vtotal_ion.rv_down - vtotal_atom.rv_down) / self.radius

    def trimmed(self, r_cut, power=TRIM_EXPONENT_DEFAULT, amplitude=1.0):
        return self.raw * trimming_theta(self.radius, r_cut, power, amplitude)

    def removed_charge_estimate(self):
        """Fit the -q/r tail of the raw SEP; sanity check (~ total charge removed)."""
        mask = (self.radius > 6.0) & (self.radius < 12.0)
        if mask.sum() < 4:
            return None
        return float(np.mean(-self.raw[mask] * self.radius[mask]))

    def write_aims_file(self, filename, species, r_cut,
                        power=TRIM_EXPONENT_DEFAULT, amplitude=1.0,
                        r_min=1e-4):
        """
        Write the trimmed SEP in the plain format read by the
        aimshalf FHI-aims patch (keyword aimshalf_potential; fhiaims/external_atomic_potential.f90):

            # comment lines
            n_points
            r_1(bohr)  V_1(Hartree)
            ...
        Points inside r_min are dropped (V_S -> finite but rV data noisy at r->0;
        the Fortran side extrapolates flat below the first point).
        """
        v = self.trimmed(r_cut, power, amplitude)
        mask = self.radius >= r_min
        r_out, v_out = self.radius[mask], v[mask]
        # keep a modest tail of explicit zeros beyond r_cut for safe splining
        stop = np.searchsorted(r_out, r_cut * 1.10)
        r_out, v_out = r_out[:stop + 1], v_out[:stop + 1]
        with open(filename, "w") as fh:
            fh.write(f"# aims_half trimmed self-energy potential\n")
            fh.write(f"# species: {species}\n")
            fh.write(f"# r_cut_bohr: {r_cut:.6f}  power: {power}  "
                     f"amplitude: {amplitude:.4f}\n")
            fh.write(f"# columns: r(bohr)  V_S(Hartree)\n")
            fh.write(f"{len(r_out)}\n")
            for r, vv in zip(r_out, v_out):
                fh.write(f"{r:22.14e} {vv:22.14e}\n")
        return filename
