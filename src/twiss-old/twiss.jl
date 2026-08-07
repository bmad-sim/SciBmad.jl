const DEFAULT_DE_MOIVRE = []
const DEFAULT_SAGAN_RUBIN = [beta_1, beta]

function twiss(
  bl::Beamline; 

  # High level customizer kwargs
  spin::Bool                = false,
  de_moivre::Bool           = false,
  at::Union{Colon, Vector}  = :,
  in_body_coordinates::Bool = false, 

  chrom::Integer = 0,
  order::Integer = 1,
  GTPSA_descriptor::Union{Descriptor,Nothing} = nothing,

  # Initial input, CO guess if periodic, initial orbit if not periodic
  delta0::Number = 0.,
  v0::Matrix     = [0. 0. 0. 0. 0. delta0], 
  
  # These guys can eventually be moved into columns/what   
  normalizing_map::Bool = false,
  RDTs::Bool            = false,

  start::Union{Integer,LineElement,Nothing} = nothing, # TODO: Nothing means compute periodic  

  a_initial::Union{Nothing,DAMap}   = nothing, # TODO, always 6D for open

  symplectic_tol=1e-8, # Tolerance below which to include damping

  # Internal, almost definitely should not be used
  _override_chrom::Bool=false,
  )

  if isnothing(start)
    v0_and_coast = co_and_coast(bl, v0)
  else
    error("Open twiss not implemented yet")
  end

  if isnothing(GTPSA_descriptor)
    storedesc = GTPSA.desc_current
    GTPSA_descriptor = Descriptor([order, order, order, order, order, order+chrom], order+chrom),
    GTPSA.desc_current = storedesc # Don't reset the global
  elseif chrom != 0 || order != 1
    @info "`GTPSA_descriptor` has been explicitly provided: ignoring `order`/`chrom` inputs"
  end

  if !v0_and_coast[2] && chrom != 0 && !_override_chrom
    error("""
    You specified `chrom`, but this beamline has synchrotron motion. Please turn off RF 
    cavities to get delta-dependent Twiss functions.
    """)
  end

  if !isnothing(a_initial) && GTPSA.getdesc(first(a_initial.v)) != GTPSA_descriptor
    error("Specified `GTPSA_descriptor` disagrees with that of `a_initial`")
  elseif !isnothing(a_initial)
    if spin && isnothing(a_initial.q)
      error("Unable to propagate spin: `a_initial` does not include spin")
    elseif !spin && !isnothing(a_initial)
      a_initial = DAMap(v0=a_initial.v0, v=a_initial.v, nv=NNF.nvars(a_initial), np=NNF.nparams(a_initial), s=a_initial.s)
    end
    GTPSA_descriptor = GTPSA.getdesc(first(a_initial.v))
  end

  # Check if output are TPSA in parameters (delta excluded)
  if GTPSA.numnn(GTPSA_descriptor) > 6
    parametric = Val{true}()
  else
    parametric = Val{false}()
  end

  # Type unstable steps:
  s, names, idxs, step_save = _twiss_assemble_locations(bl, at)
  concat, eye, zero_LF, zero_phase, zero_orbit, zero_h = _twiss_gather_types(step_save, v0_and_coast, GTPSA_descriptor, Val{spin}(), Val{RDTs}())
  table = length(step_save) == 0 ? false : true
  
  # Type stable steps:
  if concat
    return _twiss_concat(bl, eye, s, names, idxs, step_save, symplectic_tol, zero_LF, zero_phase, zero_orbit, zero_h, in_body_coordinates, Val{de_moivre}(), Val{normalizing_map}(), Val{table}())
  else
    return _twiss_noconcat(bl, eye, s, names, idxs, step_save, symplectic_tol, zero_LF, zero_phase, zero_orbit, zero_h, in_body_coordinates, Val{de_moivre}(), Val{normalizing_map}(), Val{table}())
  end
end

function co_and_coast(bl, v0)
  co_sol = find_closed_orbit(bl; v0=v0, batch=Val{false}())
  if co_sol.sol.retcode != RETCODE_SUCCESS
    error("Closed orbit finder did not converge.")
  end
  return (co_sol.v0, co_sol.coasting_beam)
end

function _twiss_assemble_locations(bl::Beamline, at::Vector)
  at_idxs = filter(x->x isa Integer, at)
  at_eles = filter(x->x isa LineElement, at)
  at_ranges = filter(x->x isa Tuple, at)
  
  stmp = Vector{Any}(undef, 0)
  names = Vector{String}(undef, 0)
  idxs = Vector{Int}(undef, 0)
  step_save = Vector{Int}(undef, 0)

  # As a guess assume length equal to number of beamline elements + 1
  # This makes the typical Twiss case hopefully faster
  n_ele = length(bl.line)
  sizehint!(stmp, n_ele+1)
  sizehint!(names, n_ele+1)
  sizehint!(idxs, n_ele+1)
  sizehint!(step_save, n_ele)

  scur = 0f0
  step_cur = 0
  for ele in bl.line
    idx = ((ele.BeamlineParams)::BeamlineParams).beamline_index
    up = (ele.UniversalParams)::UniversalParams
    name = up.name
    tm = up.tracking_method
    L = up.L
    n_steps, ds_step = BeamTracking.find_steps(tm, L)

    # Check which steps are inside any of the ranges
    found = false
    for _ in 1:n_steps
      if any(x -> x[1] <= scur < x[2], at_ranges)
        push!(stmp, scur)
        push!(names, name)
        push!(idxs, idx)
        push!(step_save, step_cur)
        found = true
      end
      step_cur += 1
      scur += ds_step
    end
    
    # If not in an s-range, check if explicitly provided (BUT ONLY AT BEGINNING!)
    # therefore need to be done at the PREVIOUS element LAST step!
    # First element must be handled specially.
    if !found && ((any(x -> x == idx, at_idxs) || any(at_eles) do x
          x == ele || (haskey(getfield(ele, :pdict), InheritParams) ? x == (getfield(ele, :pdict)[InheritParams].parent) : false)
        end
        ))
        push!(stmp, scur - ds_step*n_steps)
        push!(names, name)
        push!(idxs, idx)
        push!(step_save, step_cur-n_steps)
        #step_cur += 1
    end
  end

  # Now check if any went beyond the length of the line, in which 
  # case also save at the end of the last element.
  if any(x -> x[1] <= scur < x[2], at_ranges)
    push!(stmp, scur)
    push!(names, "END OF BEAMLINE")
    push!(idxs, -1)
    push!(step_save, step_cur)
  end

  # Now resolve type of s:
  s = typeof(scur).(stmp)

  return s, names, idxs, step_save
end

# Colon means save everywhere:
_twiss_assemble_locations(bl::Beamline, ::Colon) = _twiss_assemble_locations(bl, [(0., Inf)])

function _twiss_gather_types(step_save, v0_and_coast, GTPSA_descriptor, ::Val{spin}, ::Val{RDTs}) where {spin, RDTs}
  v0 = v0_and_coast[1]
  coasting_beam = v0_and_coast[2]

  if isnothing(GTPSA_descriptor)
    storedesc = GTPSA.desc_current
    GTPSA_descriptor = Descriptor(6,1)
    GTPSA.desc_current = storedesc # Don't reset the global
  end

  nn = GTPSA.numnn(GTPSA_descriptor)
  if nn < 6
    error("GTPSA Descriptor must have at least 6 variables for the 6D phase space coordinates")
  end

  numtype = eltype(v0)
  init = TI.InitGTPSA{GTPSA.Dynamic,Descriptor}(; dynamic_descriptor=GTPSA_descriptor)
  mo = TI.maxord(init)
  nn = TI.ndiffs(init)
  nv = 6
  np = nn-nv
  if coasting_beam
    nv -= 1
    np += 1
  end

  # If uniform truncation order, then we can concatenate maps around the ring 
  # instead of tracking twice (one to get `m`, another to propagate `a`
  desc = unsafe_load(GTPSA_descriptor.desc)
  po = desc.po
  if all(x->x == mo && (po == 0 || x == po), unsafe_wrap(Vector{UInt8}, desc.no, nn))
    concat = true
  else
    if RDTs
      error("twiss with RDTs=true requires a GTPSA Descriptor with uniform truncation order for all variables and parameters")
    end
    concat = false
  end

  # Type of the LATTICE FUNCTIONS
  if mo > 1 && (coasting_beam || nn > 6)
    zero_LF = TI.init_tps(numtype, init)
  else
    zero_LF = zero(numtype)
  end

  # Type of the PHASES
  # right now coasting beam makes phi_3 be delta-dependent 
  # even if linear. should revisit this.
  if (mo > 1 && nn > 6) || coasting_beam
    zero_phase = TI.init_tps(numtype, init)
  else
    zero_phase = zero(numtype)
  end
  
  # Type of the ORBIT
  if coasting_beam || nn > 6
    zero_orbit = TI.init_tps(numtype, init)
  else
    zero_orbit = zero(numtype)
  end

  # Value type of the RDT dict
  if RDTs
    if mo == 1
      error("
        RDTs cannot be computed using a GTPSA_descriptor with max order 1.
        Please specify a higher order GTPSA_descriptor.
      ")
    end
    if np > 0
      zero_h = TI.init_tps(numtype, init)
    else
      zero_h = zero(numtype)
    end
  else
    zero_h = nothing # Don't compute it
  end

  eye = DAMap(init=init, nv=nv, np=np, v0=view(v0, :, 1:nv), v_matrix=I, q=(spin ? I : nothing))
  return concat, eye, zero_LF, zero_phase, zero_orbit, zero_h
end

function TWISS_STATIC_CONSTS(
  a::DAMap{<:Any,<:Any,Q},
  zero_LF::T, 
  zero_phase::V, 
  zero_orbit::U, 
  zero_h::H,  
  ::Val{de_moivre}, 
  ::Val{normalizing_map}, 
  ) where {Q, T, V, U, H, de_moivre, normalizing_map}
  
  COMPUTE_TWISS = de_moivre ? compute_de_moivre : compute_sagan_rubin
  LF = !de_moivre ? twiss_tuple : de_moivre_tuple 
  LF_TABLE = !de_moivre ? twiss_table : de_moivre_table
  SCALAR_LF = TI.is_tps_type(T) isa TI.IsTPSType ? Val{false}() : Val{true}()
  SCALAR_PHASE = TI.is_tps_type(V) isa TI.IsTPSType ? Val{false}() : Val{true}()
  SCALAR_ORBIT = TI.is_tps_type(U) isa TI.IsTPSType ? Val{false}() : Val{true}()
  INCLUDE_A = normalizing_map ? at -> at : at -> nothing
  BENGTSSON = H == Nothing ? (a0, a1, m)->nothing : compute_bengtsson

  if Q == Nothing
    PROCESS_SPIN = at -> nothing
  else
    i2 = zero(a)
    NNF.setray!(i2.v; v_matrix=I)
    TI.seti!(i2.q.q2, 1, 0)
    let i2=i2
      PROCESS_SPIN = at -> begin
        n = at ∘ i2 ∘ inv(at)
        SA[n.q.q1, n.q.q2, n.q.q3]
      end
    end
  end
  # Note:
  # Descriptor(6,1) with coasting beam gives SCALAR_LF = true 
  # but SCALAR_ORBIT = false
  # In general we will canonise using SCALAR_ORBIT, and compute 
  # lattice functions using SCALAR_LF. 
  # Finally we have the phases. The phases are done during 
  # canonization, and so should have the same type as the orbit.
  if SCALAR_ORBIT isa Val{false}
    PROCESS_ORBIT = v -> begin
      StaticArrays.sacollect(SVector{6,U}, begin 
      vi = zero(v[i])
      TI.copy_tps!(vi, v[i])
      if i < 6
        TI.seti!(vi, 0, i)
      end
      vi
      end for i in 1:6)
    end
  else
    PROCESS_ORBIT = v -> StaticArrays.sacollect(SVector{6,U}, TI.scalar(v[i]) for i in 1:6)
  end
  return (; COMPUTE_TWISS, LF, LF_TABLE, SCALAR_LF, SCALAR_PHASE, INCLUDE_A, BENGTSSON, PROCESS_SPIN, PROCESS_ORBIT)
end

function _twiss_compute_row(
  twfn::T,
  damping,
  phase,
  s,
  idx,
  name,
  a,
  m_turn,
  ::Val{first}=Val{false}(),
  ) where {T,first}
  COMPUTE_TWISS = twfn.COMPUTE_TWISS 
  LF = twfn.LF 
  SCALAR_LF = twfn.SCALAR_LF 
  SCALAR_PHASE = twfn.SCALAR_PHASE 
  INCLUDE_A = twfn.INCLUDE_A 
  BENGTSSON = twfn.BENGTSSON 
  PROCESS_SPIN = twfn.PROCESS_SPIN 
  PROCESS_ORBIT = twfn.PROCESS_ORBIT

  if first
    r = canonise(a, SCALAR_PHASE; damping=damping)
  else
    r = canonise(a, SCALAR_PHASE; damping=damping, phase=phase)
  end

  a = a ∘ r
  fc = factorise(a)
  lf = LF(
    s,
    idx, 
    name, 
    SA[
      copy(phase[1]),
      copy(phase[2]),
      copy(phase[3])
    ], 
    COMPUTE_TWISS(fc.a1, SCALAR_LF), 
    PROCESS_ORBIT(fc.a0.v), 
    PROCESS_SPIN(a), 
    INCLUDE_A(a),
    BENGTSSON(fc.a0, fc.a1, m_turn)
  )
  return lf, a
end




function _twiss_tunes_and_a(m::DAMap)
  mo = NNF.maxord(m)
  a = normal(m)
  c = c_map(m) # Transform to phasor basis
  r = inv(c) ∘ inv(a) ∘ m ∘ a ∘ c
  # Need to cut highest order
  Q_x = -cutord(angle(NNF.factor_out(r.v[1], 1))/(2*pi), mo)
  Q_y = -cutord(angle(NNF.factor_out(r.v[3], 3))/(2*pi), mo)
  if NNF.nvars(m) == 5
    Q_s = real(r.v[5])
    TI.seti!(Q_s, 0, 5) # subtract time identity
  else
    Q_s = -cutord(angle(NNF.factor_out(r.v[5], 5))/(2*pi), mo)
  end
  if isnothing(m.q)
    return SA[Q_x, Q_y, Q_s], a
  else
    Q_spin = -atan(real(r.q.q2), real(r.q.q0))/pi # not two pi bc quaternion
    return SA[Q_x, Q_y, Q_s, Q_spin], a
  end
end