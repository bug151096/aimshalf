# FHI-aims hooks

aimshalf does not distribute FHI-aims code. `install_hooks.py` opens your own FHI-aims
source tree, finds an anchor line in each of five files and inserts the aimshalf lines
next to it, then copies the aimshalf module `external_atomic_potential.f90` (aimshalf's
own code, kept in this folder) into `src/`:

| File | Change |
|---|---|
| `CMakeLists.txt` | module added to library `aims4`, where its first user `read_control.f90` is built |
| `dimensions.f90` | first pass over control.in accepts the keyword (both names) |
| `read_control.f90` | the keyword `aimshalf_potential <species> <file>` is stored (the old name `external_atomic_potential` is also accepted) |
| `prepare_scf.f90` | files read, splines built, species names mapped to aims indices |
| `integrate_hamiltonian_matrix_p2.f90` | V_S added to `local_potential_parts` (Hartree + xc) at every grid point, every SCF iteration |

```bash
python3 install_hooks.py /path/to/fhi-aims.261002 --check    # dry run, writes nothing
python3 install_hooks.py /path/to/fhi-aims.261002            # install
python3 install_hooks.py /path/to/fhi-aims.261002 --remove   # restore the original files
```

All anchors are located before anything is written; if one is missing or ambiguous,
nothing is changed. Running it twice does not insert twice. A tree already edited by an
older aimshalf patch or by hand is refused: start from a pristine tree. On pristine
FHI-aims 260331.1 and 261002 the result is byte-identical to the former reference patches,
and `--remove` gives back the pristine files. Then build as usual and check the build
with `run_install_tests.sh` (this folder).

## Behaviour
- V_S at a grid point is the sum over all atom centres within r_cut, using the
  `pbc_lists` centres (periodic images included).
- Hamiltonian only: no total-energy term (DFT-1/2 total energies and forces are not
  meaningful) and no effect on the free-atom solver or the basis.
- GPU builds: the potential is evaluated on the CPU before the GPU matrix products, so
  V_S is included (not tested).
- Relativity: with `atomic_zora` V_S does not enter the ZORA kinetic term; with full
  `zora` it does, through `local_potential_parts` (not tested).
- Not added in `integrate_real_hamiltonian_matrix_components` (used only by QM
  embedding and decomposed eigenvalue output).
- The file name in control.in is read list-directed: no `/` in it (aimshalf writes
  plain names next to control.in).

## Porting to a new aims release
```bash
python3 install_hooks.py /path/to/fhi-aims.NEW --check
```
If an anchor is not found, find the equivalent line in the new release and update the
`anchor` (and, if needed, `after`) of that hook in the `HOOKS` list of `install_hooks.py`.
