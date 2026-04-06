# Experimental
function find_one_torus(
    bl::Beamline;
    v0=zeros(1,6), 
    reltol=1e-13,
    abstol=1e-13, 
    maxiter=100, 
    autodiff=KA.get_backend(v0) isa KA.GPU ? DI.AutoForwardFromPrimitive(AutoForwardDiff()) : AutoForwardDiff(),
    prep=nothing,
    coast::Union{Nothing,Bool}=nothing,
    checkconverged::Bool=true,
)
  # First do a coast check:
  if isnothing(coast)
    coast = coast_check(bl, autodiff)
  end
  
  N_particles = size(v0, 1)
  device = KA.get_backend(v0)
  batched = Val{N_particles > 1}()

  # Newton requires AoS for batching, so 
  # all residual functions 6 x N or 4 x N (coast)
  v0_cache = similar(v0, (size(v0, 2), size(v0, 1)))

  if coast
    v = similar(v0, (4, N_particles))
    v_coast = similar(v0, (4, N_particles))
    v_coast .= transpose(view(v0, :, 1:4))
    set_kernel! = set_v_coast!(device)
    sub_kernel! = sub_v!(device)
    sol = newton!(_co_res_coast!, v, v_coast, DI.Constant(bl), DI.Constant(set_kernel!), DI.Constant(sub_kernel!), DI.Cache(v0_cache), DI.Constant(transpose(v0)); reltol=reltol, abstol=abstol, maxiter=maxiter, autodiff=autodiff, prep=prep, checkconverged=Val{checkconverged}(), batched=batched)
    set_v_coast_final!(device)(v0, transpose(v_coast); ndrange=N_particles)
    KA.synchronize(device)
    @reset sol.u = v0
    sol = merge(sol, (;coast=true))
  else
    v = similar(v0, (6, N_particles))
    set_kernel! = set_v!(device)
    sub_kernel! = sub_v!(device)
    sol = newton!(_co_res!, v, transpose(v0), DI.Constant(bl), DI.Constant(set_kernel!), DI.Constant(sub_kernel!), DI.Cache(v0_cache); reltol=reltol, abstol=abstol, maxiter=maxiter, autodiff=autodiff, prep=prep, checkconverged=Val{checkconverged}(), batched=batched)
    sol = merge(sol, (;coast=false))
  end
  return sol
end