function default_solver(device, _y, _x, batchdim)
  _lx = length(_x)
  _ly = length(_y)
  if isnothing(batchdim)
    let lx=_lx, ly=_ly
      return (dx, jac, y)->(@show jac; reshape(dx, lx) .= -jac \ reshape(y, ly))
    end
  elseif batchdim == 2 # Do each serially
    _batchsize = size(_x, 2)
    _n_rows = size(_y, 1)
    _n_cols = size(_x, 1)
    let n_rows=_n_rows, n_cols=_n_cols, batchsize=_batchsize, jacsize=_n_rows*_n_cols
      return (dx, jac::SparseMatrixCSC, y)->begin
        for i in 1:batchsize
          jac_offset = (i-1)*jacsize 
          curjac = reshape(view(jac.nzval, (jac_offset+1):(jac_offset+jacsize)), (n_rows, n_cols))
          dx_offset = (i-1)*n_cols
          y_offset = (i-1)*n_rows
          view(dx, (dx_offset+1):(dx_offset+n_cols)) .= -curjac \ view(y, (y_offset+1):(y_offset+n_rows))
        end
      end
    end
  elseif batchdim == 1
    _batchsize = size(_x, 1)
    _n_rows = size(_y, 2)
    let n_rows=_n_rows, batchsize=_batchsize, xlen=length(_x), ylen=length(_y)
      return (dx, jac::SparseMatrixCSC, y)->begin
        for i in 1:batchsize
          curjac = view(reshape(jac.nzval, n_rows, :), :, i:batchsize:ylen)
          dx[i:batchsize:xlen] .= -curjac \ view(y, i:batchsize:ylen)
        end
      end
    end
  else
    error("Invalid batchdim (must be either 1, 2, or nothing)")
  end
end

"""
    newton!(f!, y, x; reltol=1e-13, abstol=1e-13,  maxiter=100, autodiff=AutoForwardDiff())

Finds roots of f!(y, x) using Newton's method. y and x will be mutated during solution.
x will contain the result.

# Arguments
- `f!`: Function that mutates y in place with the residual vector
- `y`: Residual vector
- `x`: Initial guess

# Keyword arguments
- `abstol`: Convergence absolute tolerance (default: 1e-13)
- `reltol`: Convergence relative tolerance (default: 1e-13)
- `maxiter`: Maximum number of iterations (default: 100)

Returns `NamedTuple` containing newton search results.
"""
function newton!(
  f!::Function,  # DO NOT SPECIALIZE ON FUNCTION, no need
  y::Y, 
  x::X,
  contexts::Vararg{DI.Context};
  reltol=1e-13,
  abstol=1e-13, 
  maxiter=100, 
  # On GPU need to use ForwardDiff from primitive (pushforward) for no scalar indexing
  autodiff=KA.get_backend(x) isa KA.GPU ? AutoForwardFromPrimitive(AutoForwardDiff()) : AutoForwardDiff(),
  prep=nothing, 
  batchdim::Union{Nothing,Integer}=nothing,
  solver::T=default_solver(KA.get_backend(x), y, x, batchdim), # We do specialize on the solver tho
  dx=zero.(x), # Temporary
) where {Y,X,T}
  if !isnothing(batchdim)
    if !(batchdim in (1,2))
      error("batchdim must be either 1 or 2")
    end

    # Sanity checks
    if size(x, batchdim) != size(y, batchdim)
        error("Input/output matrix size mismatch for batched newton: batched-dimension $batchdim size for 
                the input and output must be equal. Received $(size(x, batchdim)) and $(size(y, batchdim)) respectively.")
    end

    # If we are batch and the user has NOT specified an AutoSparse autodiff, set it up for them
    if !(autodiff isa AutoSparse)
      if !isnothing(prep)
        @warn "You specified a batchdim and provided AD prep, but your AD autodiff is NOT AutoSparse, which is required for batched-Newton.
               Your prep will therefore NOT be used."
        prep = nothing
      end


      batchsize = size(x, batchdim)
      otherdim = mod(batchdim, 2) + 1
      n_rows = size(y, otherdim)
      n_cols = size(x, otherdim)

      # Make it on the CPU
      nnz = batchsize * n_rows * n_cols
      rows = Vector{Int}(undef, nnz)
      cols = Vector{Int}(undef, nnz)

      rs, ri = batchdim == 1 ? (1, batchsize) : (n_rows, 1)
      cs, ci = batchdim == 1 ? (1, batchsize) : (n_cols, 1)

      idx = 1
      for i in 1:batchsize
          for r in 1:n_rows
              for c in 1:n_cols
                  rows[idx] = (i-1)*rs + (r-1)*ri + 1
                  cols[idx] = (i-1)*cs + (c-1)*ci + 1
                  idx += 1
              end
          end
      end
      d_rows = similar(y, Int, nnz)
      d_cols = similar(y, Int, nnz)
      d_mat = similar(y, Bool, nnz)
      copyto!(d_rows, rows)
      copyto!(d_cols, cols)
      d_mat .= 1

      pattern = sparse(d_rows, d_cols, d_mat, batchsize*n_rows, batchsize*n_cols)
      color = (batchdim == 1) ? repeat(1:n_cols, inner=batchsize) : repeat(1:n_cols, outer=batchsize) 
      alg = ConstantColoringAlgorithm(pattern, color; partition=:column)
      #@show color
      #        problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
      #  result = coloring(pattern, problem, GreedyColoringAlgorithm())
      #  @show row_colors(result)
#=
      # CSC for batchdim=2, CSR for batchdim=1
      if batchdim == 1
        pattern = transpose(sparse(d_cols, d_rows, d_mat, batchsize*n_cols, batchsize*n_rows))
        color = repeat(1:n_cols, inner=batchsize)
        #=
        @show color
        problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
        result = coloring(pattern, problem, GreedyColoringAlgorithm())
        @show column_colors(result)=#
        alg = ConstantColoringAlgorithm(pattern, color; partition=:column)
      else

      end
      =#
      detector = ADTypes.KnownJacobianSparsityDetector(pattern)
      autodiff = AutoSparse(autodiff; 
        sparsity_detector=detector,
        coloring_algorithm=alg,
      )
    end
  end

  if isnothing(prep)
    prep = DI.prepare_jacobian(f!, y, autodiff, x, contexts...)
  end
  if autodiff isa AutoSparse
    #=
    p = sparsity_pattern(prep)
    # Because SparseArrays is annoying and they only have API for CSC:
    if p isa Transpose
      jac = transpose(similar(transpose(p), eltype(y)))
    elseif p isa Adjoint
      jac = adjoint(similar(adjoint(p), eltype(y)))
    else
      jac = similar(p, eltype(y))
    end
    =#
    jac = similar(sparsity_pattern(prep), eltype(y))
  else
    if Y <: StaticArray && X <: StaticArray
      jac = similar(y, Size(length(Y), length(X)))
    else
      jac = similar(y, length(y), length(x))
    end
  end
  let _f! = f!, _prep = prep, _backend = autodiff
    val_and_jac!(_y, _jac, _x, _contexts...) = DI.value_and_jacobian!(_f!, _y, _jac, _prep, _backend, _x, _contexts...)
    return newton!(val_and_jac!, y, jac, x, contexts...; reltol=reltol, abstol=abstol, maxiter=maxiter, batchdim=batchdim, solver=solver, dx=dx)
  end
end

function newton!(
  val_and_jac!::Function,
  y,
  jac,
  x,
  contexts...;
  reltol=1e-13,
  abstol=1e-13, 
  maxiter=100, 
  batchdim::Union{Nothing,Integer}=nothing, 
  n_iters=nothing, # If batch, then array that should be modified in-place with the iteration when convergence reached
  converged=nothing,
  solver::T=default_solver(KA.get_backend(x), y, x, batchdim), 
  dx=zero.(x),
) where {T}
  # Setup:
  out = (; u=x, jac=jac)
  if isnothing(batchdim)
    if !isnothing(n_iters)
      @warn "You provided `n_iters`, but this is only used for batched-Newton. Non-batched Newton 
             always returns a scalar `n_iters`."
    end
    if !isnothing(converged)
      @warn "You provided `converged`, but this is only used for batched-Newton. Non-batched Newton 
             always returns a Bool `converged`."
    end
    out = merge(out, (; converged=false, n_iters=0))
    # Newton:
    dx .= 0
    for iter in 1:maxiter
      val_and_jac!(y, jac, x, contexts...)
      solver(dx, jac, y)
      if norm(y) < abstol
        @reset out.converged = true
        @reset out.n_iters = iter-1
        return out
      end
      x .= x .+ dx
      if norm(dx) < reltol*norm(x)
        @reset out.converged = true
        @reset out.n_iters = iter
        return out
      end
    end
    @reset out.n_iters=maxiter
    return out
  else
    if !isnothing(n_iters) || !isnothing(converged)
      error("Providing n_iters or converged is not implemented yet for batched-Newton")
    end
    # Newton:
    dx .= 0
    for iter in 1:maxiter
      val_and_jac!(y, jac, x, contexts...)
      solver(dx, jac, y)
      x .= x .+ dx
    end
    return out
  end
end 