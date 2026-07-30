
function _twiss_concat(
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
  ) where{de_moivre, normalizing_map, table}
  maps   = _twiss_concat_preallocate(step_save, eye)
  cb     = _twiss_concat_make_callback(step_save, maps, in_body_coordinates)
  b0     = _twiss_track(eye, (cb,), bl)
  m_turn = _twiss_concat_concatenate!(eye, b0, maps)

  tunes, a = _twiss_tunes_and_a(m_turn)

  if table
    # Check if damping:
    damping = norm(NNF.checksymp(NNF.jacobian(m_turn))) > symplectic_tol

    # Construct array for phase advance:
    phase = MVector{3}(zero(zero_phase),zero(zero_phase),zero(zero_phase))

    # Assemble the twiss evaluationg functions:
    twfn = TWISS_STATIC_CONSTS(a, zero_LF, zero_phase, zero_orbit, zero_h, Val{de_moivre}(), Val{normalizing_map}())
    
    # Push a to the first location
    a = maps[1] ∘ a

    #=
    If Bengtsson computed, we want 1-turn map evaluated at 
    this position. This is obtained by observing

    m_turn_1 =           (m_{1←3} ∘ m_{3←2} ∘ m_{2←1})
    m_turn_2 = (m_{2←1} ∘ m_{1←3} ∘ m_{3←2})
    
    m_turn_2 = m_{2←1} ∘ m_turn_1 ∘ inv(m_{2←1})

    =#
    m_turn = isnothing(zero_h) ? m_turn : (maps[1] ∘ m_turn ∘ inv(maps[1]))

    lf1, a = _twiss_compute_row(twfn, damping, phase, s[1], idxs[1], names[1], a, m_turn, Val{true}()) 

    # Construct the table and add this first row:
    len = length(maps)
    lf_table = twfn.LF_TABLE(lf1, len)
    lf_table[1] = lf1

    # Go thru all rows now:
    for i in 2:len
      a = maps[i] ∘ a
      m_turn = isnothing(zero_h) ? m_turn : (maps[i] ∘ m_turn ∘ inv(maps[i]))
      lfi, a = _twiss_compute_row(twfn, damping, phase, s[i], idxs[i], names[i], a, m_turn) 
      lf_table[i] = lfi
    end

    return make_twiss(eye, tunes, lf_table)
  else
    return make_twiss(eye, tunes, nothing)
  end
end

function _twiss_concat_preallocate(step_save, map::T) where {T<:DAMap}
  maps = Vector{T}(undef, length(step_save))
  for i in 1:length(step_save)
    if i == 1 && step_save[1] == 0
      maps[1] = one(map)
      NNF.setscalar!(maps[1], map.v0)
    else
      maps[i] = zero(map) # Preallocate
    end
  end
  return maps
end


function _twiss_concat_make_callback(_step_save, _maps, _in_body_coordinates)
  # Note: need to handle the first element differently
  if length(_step_save) > 0 && first(_step_save) == 0
    _cur_step_save_idx = 2
  else
    _cur_step_save_idx = 1
  end
  let step_save=_step_save, maps=_maps, curstep=Ref{Int}(0), cur_step_save_idx=Ref{Int}(_cur_step_save_idx), in_body_coordinates=_in_body_coordinates
    return (i, coords, cur_s, cur_t_ref, last_ds_step, last_g, transforms_out!, transforms_in!) -> begin
      curstep[] += 1
      if cur_step_save_idx[] <= length(step_save) && curstep[] == step_save[cur_step_save_idx[]] # Store the current map
        map = maps[cur_step_save_idx[]]
        if !in_body_coordinates
          transforms_out!(i, coords, cur_s, cur_t_ref)
        end
        _twiss_setmap!(map, coords)
        if !in_body_coordinates
          transforms_in!(i, coords, cur_s, cur_t_ref)
        end
        cur_step_save_idx[] += 1
      end
    end
  end
end

function _twiss_concat_concatenate!(eye, b0, maps)
  # Now we just concatenate the maps
  m_turn = eye
  i=1
  for map in maps
    m_turn = map ∘ m_turn
    i += 1
  end
  # Have to do one more now
  _twiss_setmap!(eye, b0.coords)
  if length(maps) > 0
    m_turn = eye ∘ m_turn
  end
  return m_turn
end