#!/bin/bash
# aimshalf installation tests (run inside an interactive or batch allocation).
#
# Required environment variables:
#   AIMS_PATCHED   full path to the patched aims executable
#   AIMS_SPECIES   species_defaults/defaults_2020/light directory of the SAME aims release
#   MPI_LAUNCH     MPI launcher with process count, e.g. "srun -n 8" or "mpirun -np 8"
# Optional:
#   AIMS_STOCK     full path to an unpatched aims executable of the same release
#
# Usage:  bash run_install_tests.sh        (all output goes to ./install_tests/)
set -u
: "${AIMS_PATCHED:?set AIMS_PATCHED}"; : "${AIMS_SPECIES:?set AIMS_SPECIES}"; : "${MPI_LAUNCH:?set MPI_LAUNCH}"
HERE=$(cd "$(dirname "$0")" && pwd); T=$PWD/install_tests; rm -rf "$T"; mkdir -p "$T"
pass=0; fail=0
ok(){ echo "  [PASS] $*"; pass=$((pass+1)); }; bad(){ echo "  [FAIL] $*"; fail=$((fail+1)); }
etot(){ grep "| Total energy of the DFT / Hartree-Fock s.c.f. calculation" "$1" | tail -1 | awk '{print $(NF-1)}'; }
gap(){ grep -E "ESTIMATED overall HOMO-LUMO gap|Overall HOMO-LUMO gap" "$1" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1; }

echo "== 1. Python side: aimshalf + atmhalf"
command -v aimshalf >/dev/null && ok "aimshalf on PATH: $(command -v aimshalf)" || bad "aimshalf not on PATH"
(cd "$T" && aimshalf gen-sep As -c p=0.5 --cut 3.8 > gen_sep.log 2>&1) && ok "gen-sep As p=0.5 cut=3.8 ran" || bad "gen-sep failed (see $T/gen_sep.log)"
cat "$T/gen_sep.log" | sed 's/^/        /'
SEP=$(ls "$T"/*As*cut3.8* 2>/dev/null | head -1)
[ -n "$SEP" ] && ok "SEP file written: $(basename "$SEP")" || bad "no As SEP file found in $T"
mkdir -p "$T/zero" && (cd "$T/zero" && aimshalf gen-sep As -c p=0.5 --cut 3.8 --amplitude 0 > /dev/null 2>&1)
ZERO=$(ls "$T"/zero/*As* 2>/dev/null | head -1)
[ -n "$ZERO" ] && ok "zero-amplitude SEP written" || bad "zero SEP not written"

if [ $fail -gt 0 ]; then
  echo; echo "Python side failed: fix it before the aims runs (is the venv active? did 'make atom' run?)"
  exit 1
fi

mkrun(){ # dir sepfile|none exe
  d="$T/$1"; mkdir -p "$d"; cp "$HERE/test_gaas/geometry.in" "$d/"
  cp "$HERE/test_gaas/control.in.base" "$d/control.in"
  [ "$2" != none ] && { cp "$2" "$d/As_sep.dat"; echo "aimshalf_potential As As_sep.dat" >> "$d/control.in"; }
  cat "$AIMS_SPECIES"/31_Ga_default "$AIMS_SPECIES"/33_As_default >> "$d/control.in"
  (cd "$d" && $MPI_LAUNCH "$3" > aims.out 2> aims.err)
  grep -q "Have a nice day" "$d/aims.out" && ok "$1 finished" || bad "$1 did not finish (see $d/aims.out, aims.err)"
}
echo "== 2. FHI-aims runs (GaAs, about a minute each on 8 cores)"
mkrun plain   none    "$AIMS_PATCHED"
mkrun zero    "$ZERO" "$AIMS_PATCHED"
mkrun sep     "$SEP"  "$AIMS_PATCHED"
[ -n "${AIMS_STOCK:-}" ] && mkrun stock none "$AIMS_STOCK"

echo "== 3. Checks"
grep -q "Found aimshalf potential" "$T/sep/aims.out" && ok "patched aims recognised the keyword" || bad "keyword not recognised in sep run"
E_plain=$(etot "$T/plain/aims.out"); E_zero=$(etot "$T/zero/aims.out")
[ -n "$E_plain" ] && [ "$E_plain" = "$E_zero" ] && ok "zero SEP leaves total energy unchanged ($E_plain eV)" || bad "zero SEP changed E_tot: plain=$E_plain zero=$E_zero"
if [ -n "${AIMS_STOCK:-}" ]; then
  E_stock=$(etot "$T/stock/aims.out")
  [ "$E_stock" = "$E_plain" ] && ok "patched == stock without the keyword ($E_stock eV)" || bad "patched differs from stock: $E_stock vs $E_plain"
fi
G_plain=$(gap "$T/plain/aims.out"); G_sep=$(gap "$T/sep/aims.out")
echo "        PBE gap = $G_plain eV ; PBE-1/2 (As p, cut 3.8) gap = $G_sep eV"
awk -v a="$G_plain" -v b="$G_sep" 'BEGIN{exit !(b>a+0.3)}' && ok "SEP opens the gap by more than 0.3 eV" || bad "gap did not open as expected"

echo; echo "RESULT: $pass passed, $fail failed   (outputs in $T)"
[ $fail -eq 0 ]
