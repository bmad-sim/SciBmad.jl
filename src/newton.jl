"""
    newton!(f!, y, x; reltol=1e-13, abstol=1e-13,  max_iter=100, backend=AutoForwardDiff(), check_stable=Val{false}())

Finds roots of f!(y, x) using Newton's method. y and x will be mutated during solution.
x will contain the result.

# Arguments
- `f!`: Function that mutates y in place with the residual vector
- `y`: Residual vector
- `x`: Initial guess

# Keyword arguments
- `abstol`: Convergence absolute tolerance (default: 1e-13)
- `reltol`: Convergence relative tolerance (default: 1e-13)
- `max_iter`: Maximum number of iterations (default: 100)

Returns `NamedTuple` containing newton search results.
"""
function newton!(
  f!::Function,  # DO NOT SPECIALIZE ON FUNCTION, no need
  y::Y, 
  x::X,
  contexts::Vararg{DI.Context};
  reltol=1e-13,
  abstol=1e-13, 
  max_iter=100, 
  # On GPU need to use ForwardDiff from primitive (pushforward) for no scalar indexing
  backend=KA.get_backend(x) isa KA.GPU ? AutoForwardFromPrimitive(AutoForwardDiff()) : AutoForwardDiff(),
  prep=nothing, 
  check_stable::Val{S}=Val{false}(),
  solver!::T=(dx, jac, y)->(dx .= -jac \ y), # We do specialize on the solver tho
  batchsize::Union{Integer,Nothing}=nothing, # If not nothing, then batch processing will be done
  dx=zero.(x), # Temporary
) where {Y,X,S,T}
  if !isnothing(batchsize)
    # Batch MUST have x and y stored as MATRICES where each COLUMN is one element in the batch
    # This is important because in fact this would be AoS if the batch are particles, however for 
    # Newton search with AD, there is no SIMD anyways. The biggest benefit here is that now memory-wise
    # the Jacobian is block diagonal, which means a CSC sparse jacobian layout would give us each submatrix
    # one after the other in memory. This makes it easy to then do CUDA.getrf_batched!, and maybe 
    # doing each dense matrix serially is faster than the sparse solver? We can check.
    # Sanity checks
    if size(x, 2) != size(y, 2)
        error("Input/output matrix size mismatch for batched newton: number of columns for 
                input and output must be equal, received $(size(x, 2)) and $(size(y, 2)) respectively.")
    end
    if !isinteger(size(x, 2)/batchsize)
        error("Invalid batchsize specified: size(x, 2) / batchsize = $(size(x, 2)/batchsize) which is NOT an integer")
    end

    # If we are batch and the user has NOT specified an AutoSparse backend, set it up for them
    if !(backend isa AutoSparse)
      if !isnothing(prep)
        @warn "You provided AD prep and a batchsize, but your AD backend is NOT AutoSparse, which is required for batched-Newton.
               Your prep will therefore NOT be used"
        prep = nothing
      end

      # Make it on the CPU
      n_blocks = size(x, 2)
      n_rows = size(y, 1)
      n_cols = size(x, 1)
      nnz = n_blocks * n_rows * n_cols
      rows = Vector{Int}(undef, nnz)
      cols = Vector{Int}(undef, nnz)
      idx = 1 
      for b in 0:num_blocks-1
          row_offset = b * n_rows
          col_offset = b * n_cols

          for j in 1:n_cols       
              for i in 1:n_rows
                  rows[idx] = row_offset + i
                  cols[idx] = col_offset + j
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

      # CSC by default with sparse, even with CUDA
      pattern = sparse(d_rows, d_cols, d_mat, n_blocks*n_rows, n_blocks*n_cols)

      detector = ADTypes.KnownJacobianSparsityDetector(pattern)
      color = repeat(1:n_cols, outer=n_blocks) 
      alg = ConstantColoringAlgorithm(pattern, color; partition=:column)
      backend = AutoSparse(backend; 
        sparsity_detector=detector,
        coloring_algorithm=alg,
      )
    end
  end

  if isnothing(prep)
    prep = DI.prepare_jacobian(f!, y, backend, x, contexts...)
  end
  if backend isa AutoSparse
    jac = similar(sparsity_pattern(prep), eltype(y))
  else
    if Y <: StaticArray && X <: StaticArray
      jac = similar(y, Size(length(Y), length(X)))
    else
      jac = similar(y, length(y), length(x))
    end
  end
  let _f! = f!, _prep = prep, _backend = backend
    val_and_jac!(_y, _jac, _x, _contexts) = DI.value_and_jacobian!(_f!, _y, _jac, _prep, _backend, _x, _contexts...)
    return newton!(val_and_jac!, y, jac, x, contexts...; reltol=reltol, abstol=abstol, max_iter=max_iter, check_stable=check_stable, solver!=solver!, dx=dx)
  end
end

function newton!(
  val_and_jac!::Function,
  y,
  jac,
  x,
  contexts::Vararg{DI.Context};
  reltol=1e-13,
  abstol=1e-13, 
  max_iter=100, 
  check_stable::Val{S}=Val{false}(),
  solver!::T=(dx, jac, y)->(dx .= -jac \ y), 
  dx=zero.(x),
) where {S,T}
  dx .= 0
  ly = length(y)
  fabstol = abstol*sqrt(ly)
  lx = length(x)
  for iter in 1:max_iter
    val_and_jac!(y, jac, x, contexts)
    solver!(reshape(dx, lx), jac, reshape(y, ly))
    x .= x .+ dx
    if norm(dx)< reltol*norm(x) || norm(y) < fabstol
      if S
        eg = eigen(jac)
        stable = all(t->norm(t)<=1, eg.values)
        return (;u=x, converged=true, n_iters=iter, stable=stable)
      else
        return (;u=x, converged=true, n_iters=iter)
      end
    end
  end
  if S
      return (;u=x, converged=false, n_iters=max_iter, stable=false)
  else
      return (;u=x, converged=false, n_iters=max_iter)
  end
end