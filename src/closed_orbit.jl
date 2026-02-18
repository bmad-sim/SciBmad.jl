
# Because track! is many kernels, unfortunately all must be separate kernels

# For setting before tracking
@kernel function set_v_res!(v_res, v, N_particles, ::Val{coast}) where {coast}
  i = @index(Global)
  @inbounds v_res[i + 0*N_particles] = v[i,1]
  @inbounds v_res[i + 1*N_particles] = v[i,2]
  @inbounds v_res[i + 2*N_particles] = v[i,3]
  @inbounds v_res[i + 3*N_particles] = v[i,4]
  if !coast
    @inbounds v_res[i + 4*N_particles] = v[i,5]
    @inbounds v_res[i + 5*N_particles] = v[i,6]
  end
end

# For subtracting after tracking
@kernel function sub_v_res!(v_res, v, N_particles, ::Val{coast}) where {coast}
  i = @index(Global)
  @inbounds v_res[i + 0*N_particles] -= v[i,1]
  @inbounds v_res[i + 1*N_particles] -= v[i,2]
  @inbounds v_res[i + 2*N_particles] -= v[i,3]
  @inbounds v_res[i + 3*N_particles] -= v[i,4]
  if !coast
    @inbounds v_res[i + 4*N_particles] -= v[i,5]
    @inbounds v_res[i + 5*N_particles] -= v[i,6]
  end
end

get_val_parameter(::Val{T}) where {T} = T


function _co_res!(
  v_res, 
  v, 
  p::T,
) where {T}
  bl = p[1]
  set_kernel! = p[2]
  sub_kernel! = p[3]
  coast = get_val_parameter(p[4])
  if coast
    @assert length(v_res) == size(v, 1)*4 "Incorrect size for residual vector"
  else
    @assert length(v_res) == size(v, 1)*6 "Incorrect size for residual vector"
  end
  N_particles = size(v, 1)
  b0 = Bunch(v)
  SciBmad.BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
  set_kernel!(v_res, v, N_particles, Val{coast}(); ndrange=N_particles)
  KA.synchronize(KA.get_backend(v))
  track!(b0, bl)
  sub_kernel!(v_res, v, N_particles, Val{coast}(); ndrange=N_particles)
  KA.synchronize(KA.get_backend(v))
  return v_res
end

function coast_check(bl, backend=DI.AutoForwardDiff())
  v0 = zeros(1,6)
  v = zeros(1,6)
  jac = zeros(6,6)
  set_kernel! = set_v_res!(KA.get_backend(v))
  sub_kernel! = sub_v_res!(KA.get_backend(v))
  DI.value_and_jacobian!(_co_res!, v, jac, backend, v0, DI.Constant((bl,set_kernel!,sub_kernel!,Val{false}())))
  return view(jac, 6, :) ≈ SA[0, 0, 0, 0, 0, 0]
end

# v0 is the array of initial particles
# should be equal to number of batch parameters
# or if coasting, set equal to number of coasting 
# particles WITH delta set accordingly already
function find_closed_orbit(
    bl::Beamline;
    v0=zeros(1,6), 
    reltol=1e-13,
    abstol=1e-13, 
    max_iter=100, 
    backend=DI.AutoForwardDiff(),
    prep=nothing,
    lambda=1,
    coast::Val{C}=Val{Nothing}(),
) where {C}
  if C == Nothing
    _coast = coast_check(bl, backend)
  else
    _coast = C
  end
  if _coast    
    v = similar(v0, (size(v0, 1), 4))
    set_kernel! = set_v_res!(KA.get_backend(v))
    sub_kernel! = sub_v_res!(KA.get_backend(v))
    p = (bl, set_kernel!, sub_kernel!, Val{_coast}())
    if !newton!(_co_res!, v, v0, p; reltol=reltol, abstol=abstol, max_iter=max_iter, backend=backend, prep=prep, lambda=lambda).converged
      error("Closed orbit finder not converging")
    end
  else
    v = similar(v0, (size(v0, 1), 6))
    set_kernel! = set_v_res!(KA.get_backend(v))
    sub_kernel! = sub_v_res!(KA.get_backend(v))
    p = (bl, set_kernel!, sub_kernel!, Val{_coast}())
    if !newton!(_co_res!, v, v0, p; reltol=reltol, abstol=abstol, max_iter=max_iter, backend=backend, prep=prep, lambda=lambda).converged
      error("Closed orbit finder not converging")
    end
  end
  return v0, _coast
end
#=
const CLOSED_ORBIT_FORWARDDIFF_PREP = (
  x = zeros(1,6);
  y = zeros(1,6);
  bl = Beamline([LineElement()]);
  set_kernel! = set_v_res!(KA.get_backend(x));
  sub_kernel! = sub_v_res!(KA.get_backend(x));
  p = (bl, set_kernel!, sub_kernel!, Val{false}());
  DI.prepare_jacobian(_co_res!, y, DI.AutoForwardDiff(), x, DI.Constant(p))
)

const CLOSED_ORBIT_FORWARDDIFF_COAST_PREP = (
  x = zeros(1,6);
  y = zeros(1,6);
  bl = Beamline([LineElement()]);
  set_kernel! = set_v_res!(KA.get_backend(x));
  sub_kernel! = sub_v_res!(KA.get_backend(x));
  p = (bl, set_kernel!, sub_kernel!, Val{true}());
  DI.prepare_jacobian(_co_res!, y, DI.AutoForwardDiff(), x, DI.Constant(p))
)
=#

#=
backend = get_backend(v)
kernel! = generic_kernel!(backend)
kernel!(coords, kc; ndrange=N_particle)
KernelAbstractions.synchronize(backend)

const CLOSED_ORBIT_FORWARDDIFF_PREP = (
  x = zeros(1,6);
  y = zeros(6);
  bl = Beamline([LineElement()]);
  DI.prepare_jacobian(_co_res!, y, DI.AutoForwardDiff(), x, DI.Constant(bl))
)



function track_a_particle!(coords, coords0, bl; use_KA=false, use_explicit_SIMD=false, scalar_params=true)
  coords .= coords0
  b0 = Bunch(reshape(coords, (1,6)))
  BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
  track!(b0, bl; use_KA=use_KA, use_explicit_SIMD=use_explicit_SIMD, scalar_params=scalar_params)
  if b0.coords.state[1] != 0x1
    @warn "Particle lost in tracking"
  end
  return coords
end

function coast_check(bl, backend=DI.AutoForwardDiff())
  x0 = zeros(6)
  y = zeros(6)
  jac = zeros(6, 6)
  DI.value_and_jacobian!(track_a_particle!, y, jac, backend, x0, DI.Constant(bl))
  return view(jac, 6, :) ≈ SA[0, 0, 0, 0, 0, 1]
end

function _co_res!(y, x, bl)
  track_a_particle!(y, x, bl)
  return y .= y .- x
end



function find_closed_orbit(
  bl::Beamline; 
  v0=zeros(6), 
  reltol=1e-13,
  abstol=1e-13, 
  max_iter=100, 
  backend=DI.AutoForwardDiff(),
  prep=CLOSED_ORBIT_FORWARDDIFF_PREP,
  lambda=1,
  coast::Val{C}=Val{Nothing}(),
) where {C}
  # First check if coasting, for this push a particle starting at 0 and see if
  # delta is a parameter
  v = zero(v0)
  if C == Nothing
    _coast = coast_check(bl, backend)
  else
    _coast = C
  end
  if _coast
    if !newton!(_co_res!, v, v0, bl; reltol=reltol, abstol=abstol, max_iter=max_iter, subspace=(1:4,1:4), backend=backend, prep=prep, lambda=lambda).converged
      error("Closed orbit finder not converging")
    end
  else
    if !newton!(_co_res!, v, v0, bl; reltol=reltol, abstol=abstol, max_iter=max_iter, backend=backend, prep=prep, lambda=lambda).converged
      error("Closed orbit finder not converging")
    end
  end
  return v0, _coast
end
=#