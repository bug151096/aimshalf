! synthetic stand-in (anchor lines only)
      use scf_convergence_tracker, only: initialize_analysis_history
      call get_times(time_prep, clock_time_prep)
          call libmbd_update_geometry()
      end if

      call get_times(time_prep, clock_time_prep)
