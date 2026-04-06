
# Because track! is many kernels, unfortunately all must be separate kernels

# For setting before tracking
@kernel function set_v!(v_res, v_cache, @Const(v), @Const(N_particles))
  i = @index(Global)
  @inbounds v_res[i + 0*N_particles] = v[i,1]
  @inbounds v_res[i + 1*N_particles] = v[i,2]
  @inbounds v_res[i + 2*N_particles] = v[i,3]
  @inbounds v_res[i + 3*N_particles] = v[i,4]
  @inbounds v_res[i + 4*N_particles] = v[i,5]
  @inbounds v_res[i + 5*N_particles] = v[i,6]

  @inbounds v_cache[i,1] = v[i,1]
  @inbounds v_cache[i,2] = v[i,2]
  @inbounds v_cache[i,3] = v[i,3]
  @inbounds v_cache[i,4] = v[i,4]
  @inbounds v_cache[i,5] = v[i,5]
  @inbounds v_cache[i,6] = v[i,6]
end

@kernel function set_v_coast!(v_res, v_cache, @Const(v_constant), @Const(v_coast), @Const(N_particles))
  i = @index(Global)

  @inbounds v_cache[i,5] = v_constant[i,5]
  @inbounds v_cache[i,6] = v_constant[i,6]

  @inbounds v_cache[i,1] = v_coast[i,1]
  @inbounds v_cache[i,2] = v_coast[i,2]
  @inbounds v_cache[i,3] = v_coast[i,3]
  @inbounds v_cache[i,4] = v_coast[i,4]

  @inbounds v_res[i + 0*N_particles] = v_coast[i,1]
  @inbounds v_res[i + 1*N_particles] = v_coast[i,2]
  @inbounds v_res[i + 2*N_particles] = v_coast[i,3]
  @inbounds v_res[i + 3*N_particles] = v_coast[i,4]
end

@kernel function set_v_coast_final!(v0, v_coast)
  i = @index(Global)
  @inbounds v0[i,1] = v_coast[i,1]
  @inbounds v0[i,2] = v_coast[i,2]
  @inbounds v0[i,3] = v_coast[i,3]
  @inbounds v0[i,4] = v_coast[i,4]
end

# For subtracting after tracking
# If a particle is lost, it should set that particle's residual to Inf
@kernel function sub_v!(v_res, v, state, N_particles, ::Val{coast}) where {coast}
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

function _co_res!(
  v_res, 
  v, 
  bl::Beamline,
  set_kernel!,
  sub_kernel!,
  v_cache,
)
  N_particles = size(v, 1)
  @assert length(v_res) == N_particles*6 "Incorrect size for residual vector"
  b0 = Bunch(v_cache)
  SciBmad.BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
  set_kernel!(v_res, v_cache, v, N_particles; ndrange=N_particles)
  KA.synchronize(KA.get_backend(v))
  track!(b0, bl, scalar_params=true)
  sub_kernel!(v_res, v_cache, b0.coords.state, N_particles, Val{false}(); ndrange=N_particles)
  KA.synchronize(KA.get_backend(v))
  return v_res
end

function _co_res_coast!(
  v_res, 
  v_coast,
  bl::Beamline,
  set_kernel!,
  sub_kernel!,
  v_cache,
  v_constant
)
  N_particles = size(v_coast, 1)
  @assert length(v_res) == N_particles*4 "Incorrect size for residual vector"
  @assert N_particles == size(v_cache, 1) "Incorrect size for particle cache array given v_coast input"
  @assert N_particles == size(v_constant, 1) "Incorrect size for particle constant array given v_coast input"
  b0 = Bunch(v_cache)
  SciBmad.BTBL.check_bl_bunch!(bl, b0, false) # Do not notify  
  set_kernel!(v_res, v_cache, v_constant, v_coast, N_particles; ndrange=N_particles)
  KA.synchronize(KA.get_backend(v_cache))
  track!(b0, bl, scalar_params=true)
  sub_kernel!(v_res, v_cache, b0.coords.state, N_particles, Val{true}(); ndrange=N_particles)
  KA.synchronize(KA.get_backend(v_cache))
  return v_res
end

function coast_check(bl, autodiff=AutoForwardDiff())
  v0 = zeros(1,6)
  v = zeros(1,6)
  v_cache = copy(v0)
  jac = zeros(6,6)
  set_kernel! = set_v!(KA.get_backend(v))
  sub_kernel! = sub_v!(KA.get_backend(v))
  DI.value_and_jacobian!(_co_res!, v, jac, autodiff, v0, DI.Constant(bl), DI.Constant(set_kernel!), DI.Constant(sub_kernel!), DI.Cache(v_cache))
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
    maxiter=100, 
    autodiff=KA.get_backend(v0) isa KA.GPU ? DI.AutoForwardFromPrimitive(AutoForwardDiff()) : AutoForwardDiff(),
    prep=nothing,
    coast=coast_check(bl, autodiff),
)
  N_particles = size(v0, 1)
  device = KA.get_backend(v0)
  v0_cache = similar(v0)
  batchdim = N_particles > 1 ? 1 : nothing

  if coast
    v = similar(v0, (N_particles, 4))
    v_coast = similar(v0, (N_particles, 4))
    v_coast .= view(v0, :, 1:4) 
    set_kernel! = set_v_coast!(device)
    sub_kernel! = sub_v!(device)
    sol = newton!(_co_res_coast!, v, v_coast, DI.Constant(bl), DI.Constant(set_kernel!), DI.Constant(sub_kernel!), DI.Cache(v0_cache), DI.Constant(v0); reltol=reltol, abstol=abstol, maxiter=maxiter, autodiff=autodiff, prep=prep, batchdim=batchdim)
    set_v_coast_final!(device)(v0, v_coast; ndrange=N_particles)
    KA.synchronize(device)
    @reset sol.u = v0
    sol = merge(sol, (;coast=true))
  else
    v = similar(v0)
    set_kernel! = set_v!(device)
    sub_kernel! = sub_v!(device)
    sol = newton!(_co_res!, v, v0, DI.Constant(bl), DI.Constant(set_kernel!), DI.Constant(sub_kernel!), DI.Cache(v0_cache); reltol=reltol, abstol=abstol, maxiter=maxiter, autodiff=autodiff, prep=prep, batchdim=batchdim)
    sol = merge(sol, (;coast=false))
  end
  return sol
end