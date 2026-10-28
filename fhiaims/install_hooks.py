#!/usr/bin/env python3
"""
install_hooks.py -- add the aimshalf hooks to YOUR copy of the FHI-aims source.

aimshalf does not distribute any FHI-aims code. This script opens the FHI-aims
source tree you already have (licensed copy), finds nine anchor lines in five
files and inserts the aimshalf lines next to them, then copies the aimshalf
module external_atomic_potential.f90 (aimshalf's own code) into src/.

    python3 fhiaims/install_hooks.py /path/to/fhi-aims.261002            # install
    python3 fhiaims/install_hooks.py /path/to/fhi-aims.261002 --check    # dry run
    python3 fhiaims/install_hooks.py /path/to/fhi-aims.261002 --remove   # undo

All anchors are located before anything is written: if one is missing or
ambiguous, nothing is changed. Running it twice does not insert twice.
--remove restores the files exactly as they were. Then build FHI-aims as usual.

Tested on pristine FHI-aims 260331.1 and 261002 (result identical to the
reference patch). Standard library only; Python >= 3.6.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

MODULE = "external_atomic_potential.f90"
HERE = Path(__file__).resolve().parent

# Each hook: file, anchor (a line, compared without leading/trailing blanks),
# 'after' (an earlier line the anchor must follow; picks the right occurrence),
# where ('before'/'after' the anchor) and the lines to insert.
HOOKS = [
    dict(file="CMakeLists.txt", name="module in library aims4",
         after=r"target_sources\(aims4 PRIVATE", anchor=r"read_control\.f90", where="after",
         lines=["  external_atomic_potential.f90"]),
    dict(file="dimensions.f90", name="keyword accepted in the first pass",
         anchor=r'elseif \(desc_str\.eq\."force_correction"\) then', where="before",
         lines=['            elseif (desc_str.eq."aimshalf_potential" .or. desc_str.eq."external_atomic_potential") then',
                "               read(inputline,*,end=88,err=99) desc_str, desc_str, desc_str"]),
    dict(file="read_control.f90", name="use statement",
         anchor=r"use dftd3_interface, only: set_d3_xc_params", where="after",
         lines=["   use external_atomic_potential"]),
    dict(file="read_control.f90", name="declarations",
         anchor=r"character\(len=8\) :: myid_string", where="after",
         lines=["   character(len=64)  :: aap_species_str",
                "   character(len=256) :: aap_file_str"]),
    dict(file="read_control.f90", name="keyword aimshalf_potential",
         anchor=r"case \('xc_pre'\)", where="before",
         lines=["         case ('aimshalf_potential', 'external_atomic_potential')",
                "            read (inputline, *, end=88, err=99) desc_str, aap_species_str, aap_file_str",
                "            call external_atomic_potential_parse_line(aap_species_str, aap_file_str)",
                "            write(info_str,'(2X,4A)') &",
                "              \"Found aimshalf potential (DFT-1/2) for species '\", &",
                "              trim(aap_species_str), \"' in file: \", trim(aap_file_str)",
                "            call localorb_info(info_str)"]),
    dict(file="prepare_scf.f90", name="use statement",
         anchor=r"use scf_convergence_tracker, only: initialize_analysis_history", where="after",
         lines=["      use external_atomic_potential"]),
    dict(file="prepare_scf.f90", name="read potential files, map species",
         after=r"call libmbd_update_geometry\(\)", anchor=r"call get_times\(time_prep, clock_time_prep\)",
         where="before",
         lines=["      ! DFT-1/2: read tabulated self-energy potential files and build splines",
                "      call external_atomic_potential_initialize()",
                "      ! DFT-1/2: map corrected species names to aims species indices",
                "      if (external_atomic_potential_enabled()) then",
                "         do i_species = 1, n_species, 1",
                "            call external_atomic_potential_set_species_index( &",
                "                 species_name(i_species), i_species)",
                "         end do",
                "      end if",
                ""]),
    dict(file="integrate_hamiltonian_matrix_p2.f90", name="use statement",
         anchor=r"use pbc_lists", where="after",
         lines=["  use external_atomic_potential"]),
    dict(file="integrate_hamiltonian_matrix_p2.f90", name="declaration",
         anchor=r"real\*8 coord_current\(3\)", where="after",
         lines=["  real*8 :: v_sep_point"]),
    dict(file="integrate_hamiltonian_matrix_p2.f90", name="SEP added to the local potential",
         anchor=r"local_potential_parts\(i_spin\) =  hartree_potential\(i_full_points\) \+ "
                r"local_xc_derivs\(i_spin,i_point\) \+ vdw_potential\(i_full_points\)",
         where="after",
         lines=["                   !=== DFT-1/2: add trimmed self-energy potential (spin-independent) ===",
                "                   if (i_spin == 1) then",
                "                      call add_external_atomic_potential_point(coord_current, &",
                "                           n_centers, coords_center, species_center, v_sep_point)",
                "                   end if",
                "                   local_potential_parts(i_spin) = local_potential_parts(i_spin) + v_sep_point",
                "                   !=== end DFT-1/2 ==="]),
]


class HookError(RuntimeError):
    pass


def _match(line, rx):
    return re.fullmatch(rx, line.strip()) is not None


def _locate(lines, hook):
    """Index of the anchor line (the first after `after`, if given)."""
    start = 0
    if hook.get("after"):
        hits = [i for i, l in enumerate(lines) if _match(l, hook["after"])]
        if not hits:
            raise HookError(f"{hook['file']}: line '{hook['after']}' not found")
        start = hits[0] + 1
    hits = [i for i in range(start, len(lines)) if _match(lines[i], hook["anchor"])]
    if not hits:
        raise HookError(f"{hook['file']}: anchor for '{hook['name']}' not found")
    if not hook.get("after") and len(hits) > 1:
        raise HookError(f"{hook['file']}: anchor for '{hook['name']}' found {len(hits)} times")
    return hits[0]


def _block_at(lines, i, block):
    return lines[i:i + len(block)] == block


def _installed(lines, hook):
    a = _locate(lines, hook)
    n = len(hook["lines"])
    if hook["where"] == "after":
        return _block_at(lines, a + 1, hook["lines"])
    return a >= n and _block_at(lines, a - n, hook["lines"])


def plan(src, remove=False):
    """New contents of every file: {path: (old_text, new_text, [messages])}."""
    out = {}
    for hook in HOOKS:
        path = src / hook["file"]
        if not path.is_file():
            raise HookError(f"{path} not found: is this the root of an FHI-aims source tree?")
        old, new, msgs = out.get(path, (None, None, []))
        if old is None:
            old = path.read_text()
            new = old
        lines = new.split("\n")
        done = _installed(lines, hook)
        a = _locate(lines, hook)
        n = len(hook["lines"])
        if remove:
            if done:
                cut = a + 1 if hook["where"] == "after" else a - n
                del lines[cut:cut + n]
                msgs.append(f"  - {hook['file']}: {hook['name']} removed")
            else:
                msgs.append(f"    {hook['file']}: {hook['name']} not present")
        else:
            if done:
                msgs.append(f"    {hook['file']}: {hook['name']} already present")
            else:
                at = a + 1 if hook["where"] == "after" else a
                lines[at:at] = hook["lines"]
                msgs.append(f"  + {hook['file']}: {hook['name']} (line {at + 1})")
        out[path] = (old, "\n".join(lines), msgs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aims_dir", help="root of your FHI-aims source tree (contains src/)")
    ap.add_argument("--check", action="store_true", help="only report, change nothing")
    ap.add_argument("--remove", action="store_true", help="remove the aimshalf hooks and module")
    a = ap.parse_args(argv)
    src = Path(a.aims_dir).expanduser().resolve() / "src"
    module_src, module_dst = HERE / MODULE, src / MODULE
    try:
        changes = plan(src, remove=a.remove)
        if not a.remove:
            # a pristine FHI-aims file never mentions the module; if it does but a hook
            # is still missing, the tree was edited by hand or by an older patch
            for path, (old, new, _) in changes.items():
                if new != old and "external_atomic_potential" in old:
                    raise HookError(f"{path.name} was already modified (older aimshalf patch or "
                                    "manual edit); start again from a pristine FHI-aims source tree")
    except HookError as err:
        sys.exit(f"install_hooks: {err}\nNothing was changed.")
    print(f"FHI-aims source: {src}")
    for _, (_, _, msgs) in changes.items():
        for m in msgs:
            print(m)
    if a.remove:
        print(f"  - src/{MODULE} {'removed' if module_dst.exists() else 'not present'}")
    else:
        same = module_dst.exists() and module_dst.read_bytes() == module_src.read_bytes()
        print(f"  {'  ' if same else '+ '}src/{MODULE} {'already present' if same else 'copied from aimshalf'}")
    if a.check:
        print("--check: nothing was written.")
        return
    for path, (old, new, _) in changes.items():
        if new != old:
            path.write_text(new)
    if a.remove:
        if module_dst.exists():
            module_dst.unlink()
        print("aimshalf hooks removed.")
    else:
        shutil.copyfile(module_src, module_dst)
        print("aimshalf hooks installed. Now build FHI-aims as usual.")


if __name__ == "__main__":
    main()
