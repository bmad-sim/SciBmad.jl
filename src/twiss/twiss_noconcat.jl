function _twiss_noconcat(
  bl,
  eye, 
  s, 
  names, 
  idxs, 
  step_save, 
  symplectic_tol,
  zero_LF, 
  zero_phase, 
  zero_orbit, 
  zero_h, 
  in_body_coordinates,
  ::Val{de_moivre},
  ::Val{normalizing_map},
  ::Val{table}
  ) where {de_moivre, normalizing_map, table}

  # For noconcat, first just track around the ring to get the 1-turn map
  b0 = _twiss_track(eye, (), bl)
  m_turn = zero(eye)
  _twiss_setmap!(m_turn, b0.coords)

  tunes, a = _twiss_tunes_and_a(m_turn)

  if table
    if !isnothing(zero_h)
      error("twiss with RDTs=true is incompatible with _twiss_noconcat")
    end

    # Check if damping:
    damping = norm(NNF.checksymp(NNF.jacobian(m_turn))) > symplectic_tol
    
    # Construct array for phase advance:
    phase = MVector{3}(zero(zero_phase),zero(zero_phase),zero(zero_phase))

    # Assemble the twiss evaluationg functions:
    twfn = TWISS_STATIC_CONSTS(a, zero_LF, zero_phase, zero_orbit, zero_h, Val{de_moivre}(), Val{normalizing_map}())
    
    #=
    With noconcat, we will do one evaluation of the lattice functions so that we can construct 
    the table, which will be put in the closure to track along with the beam. We only add this 
    if the 0th position is included in step_save.
    =#

    lf1, a = _twiss_compute_row(twfn, damping, phase, s[1], idxs[1], names[1], a, m_turn, Val{true}()) 
    lf_table = twfn.LF_TABLE(lf1, length(s))
    if first(step_save) == 0
      lf_table[1] = lf1
      initial_step_save_idx = 2
    else
      initial_step_save_idx = 1
    end

    # Make the callback
    cb = _twiss_noconcat_make_callback(twfn, step_save, lf_table, initial_step_save_idx, in_body_coordinates, damping, phase, a, m_turn)

    # Put `a` in the arrays in bunch `b0`
    for i in 1:6
      TI.copy!(b0.coords.v[i], a.v[i])
    end
    if !isnothing(a.q)
      for i in 1:4
        TI.copy!(b0.coords.q[i], a.q[i])
      end
    end

    # Add callback into new bunch and track. This will now fill the table
    bt = Bunch(v=b0.coords.v, q=b0.coords.q, callbacks=(cb,))
    BTBL.check_bl_bunch!(bt, bl, false) # Do not notify
    track!(bt, bl)

    return Twiss(NNF.nvars(m_turn) == 5, tunes, lf_table)
  else
    return Twiss(NNF.nvars(m_turn) == 5, tunes, nothing)
  end
end

function _twiss_noconcat_make_callback(
  _twfn, 
  _step_save, 
  _lf_table, 
  initial_step_save_idx, 
  _in_body_coordinates, 
  _damping, 
  _phase,
  _a,
  _m_turn, 
  )
  # stupid let block for the stupid compiler for the closure:
  let twfn=_twfn, step_save=_step_save, lf_table=_lf_table, curstep=curstep=Ref{Int}(0), cur_step_save_idx=Ref{Int}(initial_step_save_idx), 
    in_body_coordinates=_in_body_coordinates, damping=_damping, phase=_phase, a=_a, m_turn=_m_turn
    return (i, coords, cur_s, cur_t_ref, last_ds_step, last_g, transforms_out!, transforms_in!) -> begin
      curstep[] += 1
      if cur_step_save_idx[] <= length(step_save) && curstep[] == step_save[cur_step_save_idx[]]
        if !in_body_coordinates
          transforms_out!(i, coords, cur_s, cur_t_ref)
        end
        j = cur_step_save_idx[] 
        _twiss_setmap!(a, coords)
        lfj, aj = _twiss_compute_row(twfn, damping, phase, s[j], idxs[j], names[j], a, m_turn)
        lf_table[j] = lfj
        for k in 1:6
          TI.copy!(coords.v[k], aj.v[k])
        end
        if !isnothing(a.q)
          for k in 1:4
            TI.copy!(coords.q[k], aj.q[k])
          end
        end
        if !in_body_coordinates
          transforms_in!(i, coords, cur_s, cur_t_ref)
        end
        cur_step_save_idx[] += 1
      end
    end
  end
end