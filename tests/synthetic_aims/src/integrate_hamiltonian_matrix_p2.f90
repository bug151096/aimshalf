! synthetic stand-in (anchor lines only)
  use pbc_lists
  real*8 coord_current(3)
                   local_potential_parts(i_spin) =  hartree_potential(i_full_points) + local_xc_derivs(i_spin,i_point) + vdw_potential(i_full_points)
