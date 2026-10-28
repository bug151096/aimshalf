!****h* FHI-aims/external_atomic_potential
!  NAME
!    external_atomic_potential  --  reference module for the aims_half
!    DFT-1/2 hook. Reads, per species, a tabulated short-ranged radial
!    potential V_S(r) [Hartree vs bohr] and adds it, as a superposition
!    over the corresponding atomic sites, to the local KS potential on
!    the real-space integration grid at every SCF step.
!
!  DESIGN
!    * All DFT-1/2 physics (half-ionization, trimming Theta(r)) lives in
!      the Python CLI; the file read here is already trimmed and exactly
!      zero beyond r_cut. This module is therefore a *generic*
!      "external atom-centered local potential" feature.
!    * Applied to the POTENTIAL ONLY: do NOT add any corresponding term
!      to the total-energy expression -- DFT-1/2 total energies/forces
!      are not variational/meaningful. Keep geometries from plain DFT.
!    * Must NOT enter the on-site free-atom solver / basis generation:
!      hook after basis + free-atom superposition are built.
!
!  INTEGRATION (see INTEGRATION.md for the step-by-step)
!    1. read_control.f90 : parse
!           aimshalf_potential <species_symbol> <filename>
!       (one line per corrected species), store via
!       external_atomic_potential_parse_line().
!    2. After species data are known, call
!       external_atomic_potential_initialize().
!    3. In the routine that assembles the total local potential on the
!       integration grid batches (the same place where the sum of
!       free-atom potentials / hartree potential / v_xc are combined
!       into v_hamiltonian for integrate_hamiltonian_matrix), call
!           call add_external_atomic_potential(n_points, points, v_loc)
!       once per batch, AFTER v_loc is otherwise complete.
!    4. Periodic systems: loop over relevant periodic images. Because
!       V_S(r>r_cut)=0 identically, only images with
!       |r_point - R_atom - T| < r_cut contribute; reuse aims' existing
!       neighbor/atom lists for grid batches if available.
!
!  UNITS: bohr / Hartree throughout (aims internal units).
!******
module external_atomic_potential

  implicit none
  private
  
  public :: external_atomic_potential_set_species_index
  public :: add_external_atomic_potential_point
  public :: external_atomic_potential_parse_line
  public :: external_atomic_potential_initialize
  public :: external_atomic_potential_enabled
  public :: add_external_atomic_potential

  integer, parameter :: dp = kind(1.d0)
  integer, parameter :: max_species_pot = 20

  type :: radial_potential
     character(len=40)  :: species_name = ''
     character(len=400) :: filename     = ''
     integer            :: species_index = -1
     integer            :: n_grid       = 0
     real(dp), allocatable :: r(:), v(:), d2v(:)   ! d2v: spline moments
     real(dp)           :: r_cut = 0.d0
  end type radial_potential

  type(radial_potential), save :: pot(max_species_pot)
  integer, save :: n_pot = 0
  logical, save :: initialized = .false.

contains

    !----------------------------------------- map species name -> aims index
  subroutine external_atomic_potential_set_species_index(name, idx)
    character(len=*), intent(in) :: name
    integer,          intent(in) :: idx
    integer :: i
    do i = 1, n_pot
       if (trim(pot(i)%species_name) == trim(adjustl(name))) then
          pot(i)%species_index = idx
       end if
    end do
  end subroutine external_atomic_potential_set_species_index

  !----------------------------------------- SEP at one point over all centers
  ! coord      : grid-point coordinate (bohr), already in the zeroth cell
  ! n_centers  : number of atom centers incl. periodic images (pbc_lists)
  ! center_coords(3, n_centers), center_species(n_centers)
  ! v_out      : summed V_S (Hartree) to be ADDED to the local potential
  subroutine add_external_atomic_potential_point(coord, n_centers_in, &
                                     center_coords, center_species, v_out)
    real(dp), intent(in)  :: coord(3)
    integer,  intent(in)  :: n_centers_in
    real(dp), intent(in)  :: center_coords(3, n_centers_in)
    integer,  intent(in)  :: center_species(n_centers_in)
    real(dp), intent(out) :: v_out
    integer  :: ic, ip
    real(dp) :: d(3), r
    v_out = 0.d0
    if (.not. initialized) return
    do ic = 1, n_centers_in
       ip = pot_index_for_species(center_species(ic))
       if (ip == 0) cycle
       d = coord - center_coords(:, ic)
       r = sqrt(sum(d*d))
       if (r >= pot(ip)%r_cut) cycle
       v_out = v_out + eval_radial(pot(ip), r)
    end do
  end subroutine add_external_atomic_potential_point

  integer function pot_index_for_species(sp_idx)
    integer, intent(in) :: sp_idx
    integer :: i
    pot_index_for_species = 0
    do i = 1, n_pot
       if (pot(i)%species_index == sp_idx) then
          pot_index_for_species = i
          return
       end if
    end do
  end function pot_index_for_species

  !--------------------------------------------------------------- parse
  subroutine external_atomic_potential_parse_line(species_name, filename)
    character(len=*), intent(in) :: species_name, filename
    n_pot = n_pot + 1
    if (n_pot > max_species_pot) then
       stop 'external_atomic_potential: too many corrected species'
    end if
    pot(n_pot)%species_name = trim(adjustl(species_name))
    pot(n_pot)%filename     = trim(adjustl(filename))
  end subroutine external_atomic_potential_parse_line

  !---------------------------------------------------------- initialize
  subroutine external_atomic_potential_initialize()
    integer :: i, j, ios, unit_no
    character(len=400) :: line
    do i = 1, n_pot
       open(newunit=unit_no, file=pot(i)%filename, status='old', &
            action='read', iostat=ios)
       if (ios /= 0) stop 'external_atomic_potential: cannot open file'
       do  ! skip comment header
          read(unit_no, '(A)', iostat=ios) line
          if (ios /= 0) stop 'external_atomic_potential: bad file'
          if (line(1:1) /= '#') exit
       end do
       read(line, *) pot(i)%n_grid
       allocate(pot(i)%r(pot(i)%n_grid), pot(i)%v(pot(i)%n_grid), &
                pot(i)%d2v(pot(i)%n_grid))
       do j = 1, pot(i)%n_grid
          read(unit_no, *) pot(i)%r(j), pot(i)%v(j)
       end do
       close(unit_no)
       pot(i)%r_cut = pot(i)%r(pot(i)%n_grid)
       call spline_natural(pot(i)%n_grid, pot(i)%r, pot(i)%v, pot(i)%d2v)
    end do
    initialized = (n_pot > 0)
  end subroutine external_atomic_potential_initialize

  logical function external_atomic_potential_enabled()
    external_atomic_potential_enabled = initialized
  end function external_atomic_potential_enabled

  !--------------------------------------------------------------- apply
  ! points(3, n_points): integration-grid point coordinates (bohr).
  ! v_loc(n_points): local potential to be incremented (Hartree).
  !
  ! NOTE: coords_of_atom(:, i_atom) and species_name_of_atom(i_atom) are
  ! placeholders -- wire them to aims' geometry module (coords /
  ! species(species_name)) and, for periodic systems, wrap the atom loop
  ! in a loop over periodic images within r_cut (see INTEGRATION.md).
  subroutine add_external_atomic_potential(n_points, points, v_loc, &
                                           n_atoms, atom_coords, atom_species_name)
    integer,  intent(in)    :: n_points, n_atoms
    real(dp), intent(in)    :: points(3, n_points)
    real(dp), intent(in)    :: atom_coords(3, n_atoms)
    character(len=*), intent(in) :: atom_species_name(n_atoms)
    real(dp), intent(inout) :: v_loc(n_points)

    integer  :: ip, ia, is
    real(dp) :: d(3), r

    if (.not. initialized) return
    do ia = 1, n_atoms
       is = species_pot_index(atom_species_name(ia))
       if (is == 0) cycle
       do ip = 1, n_points
          d = points(:, ip) - atom_coords(:, ia)
          r = sqrt(sum(d*d))
          if (r >= pot(is)%r_cut) cycle
          v_loc(ip) = v_loc(ip) + eval_radial(pot(is), r)
       end do
    end do
  end subroutine add_external_atomic_potential

  !------------------------------------------------------------- helpers
  integer function species_pot_index(name)
    character(len=*), intent(in) :: name
    integer :: i
    species_pot_index = 0
    do i = 1, n_pot
       if (trim(pot(i)%species_name) == trim(adjustl(name))) then
          species_pot_index = i
          return
       end if
    end do
  end function species_pot_index

  real(dp) function eval_radial(p, r)
    type(radial_potential), intent(in) :: p
    real(dp), intent(in) :: r
    integer :: lo, hi, mid
    real(dp) :: h, a, b
    if (r <= p%r(1)) then           ! flat extrapolation inside first point
       eval_radial = p%v(1)
       return
    end if
    lo = 1; hi = p%n_grid
    do while (hi - lo > 1)          ! binary search
       mid = (lo + hi) / 2
       if (p%r(mid) > r) then
          hi = mid
       else
          lo = mid
       end if
    end do
    h = p%r(hi) - p%r(lo)
    a = (p%r(hi) - r) / h
    b = (r - p%r(lo)) / h
    eval_radial = a*p%v(lo) + b*p%v(hi) + &
         ((a**3 - a)*p%d2v(lo) + (b**3 - b)*p%d2v(hi)) * (h*h) / 6.d0
  end function eval_radial

  subroutine spline_natural(n, x, y, y2)
    integer,  intent(in)  :: n
    real(dp), intent(in)  :: x(n), y(n)
    real(dp), intent(out) :: y2(n)
    real(dp) :: u(n), sig, pp
    integer  :: i
    y2(1) = 0.d0; u(1) = 0.d0
    do i = 2, n - 1
       sig = (x(i) - x(i-1)) / (x(i+1) - x(i-1))
       pp  = sig * y2(i-1) + 2.d0
       y2(i) = (sig - 1.d0) / pp
       u(i) = (6.d0 * ((y(i+1)-y(i))/(x(i+1)-x(i)) - &
               (y(i)-y(i-1))/(x(i)-x(i-1))) / (x(i+1)-x(i-1)) - sig*u(i-1)) / pp
    end do
    y2(n) = 0.d0
    do i = n - 1, 1, -1
       y2(i) = y2(i) * y2(i+1) + u(i)
    end do
  end subroutine spline_natural

end module external_atomic_potential
