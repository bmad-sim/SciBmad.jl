module Experimental
using ..SciBmad, FundamentalFrequencies

function find_one_torus(
    bl::Beamline,
    mode::Integer,
    qp=[0 0];
    abstol=1e-12,     # abstol for the Newton search
    reltol=1e-12,     # reltol for the Newton search
    maxiter=100,      # Max number of Newton iterations to perform to get the 1-torus
    difftol=1e-4,     # Tolerance below which two frequencies found are considered the same, really dependent on number of turns
    n_frequencies=20, # Number of frequencies to compute in the NAFF
    order=3,          # Order of frequencies to eliminate in cross check
    fdmode=:central,  # Either :forward or :central
    n_turns=500,      # Number of turns to track each iteration
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



end