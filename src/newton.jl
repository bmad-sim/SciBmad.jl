default_solver(::KA.CPU, batchsize) = (dx, jac, y)->(dx .= -jac \ y)

"""
    newton!(f!, y, x; reltol=1e-13, abstol=1e-13,  maxiter=100, autodiff=AutoForwardDiff(), checkstable=Val{false}())

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
  checkstable::Val{_checkstable}=Val{false}(),
  batchsize::Union{Integer,Nothing}=nothing, # If not nothing, then batch processing will be done
  solver::T=default_solver(KA.get_backend(x), batchsize), # We do specialize on the solver tho
  dx=zero.(x), # Temporary
) where {Y,X,_checkstable,T}
  if !isnothing(batchsize)
    # Batch MUST have x and y stored as MATRICES where each COLUMN is one element in the batch
    # This makes the Jacobian block diagonal, which means a CSC sparse jacobian layout would give us 
    # each submatrix one after the other in memory. This makes it easy to then do CUDA.getrf_batched!, 
    # and maybe doing each dense matrix serially is faster than the sparse solver? We can check.

    # Note that if you still want SoA layout, you can allocate as SoA and transpose. Then the cost 
    # will be WRITING to the Jacobian matrix, but not in EVALUATING the Jacobian matrix if the function 
    # is accelerated by SoA
    # Sanity checks
    if size(x, 2) != size(y, 2)
        error("Input/output matrix size mismatch for batched newton: number of columns for 
                input and output must be equal, received $(size(x, 2)) and $(size(y, 2)) respectively.")
    end
    if !isinteger(size(x, 2)/batchsize)
        error("Invalid batchsize specified: size(x, 2) / batchsize = $(size(x, 2)/batchsize) which is NOT an integer")
    end

    # If we are batch and the user has NOT specified an AutoSparse autodiff, set it up for them
    if !(autodiff isa AutoSparse)
      if !isnothing(prep)
        @warn "You provided AD prep and a batchsize, but your AD autodiff is NOT AutoSparse, which is required for batched-Newton.
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
      for b in 0:n_blocks-1
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
    jac = similar(sparsity_pattern(prep), eltype(y))
  else
    if Y <: StaticArray && X <: StaticArray
      jac = similar(y, Size(length(Y), length(X)))
    else
      jac = similar(y, length(y), length(x))
    end
  end
  let _f! = f!, _prep = prep, _backend = autodiff
    val_and_jac!(_y, _jac, _x, _contexts) = DI.value_and_jacobian!(_f!, _y, _jac, _prep, _backend, _x, _contexts...)
    return newton!(val_and_jac!, y, jac, x, contexts...; reltol=reltol, abstol=abstol, maxiter=maxiter, checkstable=checkstable, batchsize=batchsize, solver=solver, dx=dx)
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
  maxiter=100, 
  checkstable::Val{_checkstable}=Val{false}(),
  solver::T=default_solver(KA.get_backend(x), batchsize), 
  batchsize::Union{Integer,Nothing}=nothing, 
  dx=zero.(x),
) where {_checkstable,T}
  dx .= 0
  lx = length(x)
  ly = length(y)
  for iter in 1:maxiter
    val_and_jac!(y, jac, x, contexts)
    solver(reshape(dx, lx), jac, reshape(y, ly), batchsize)
    x .= x .+ dx
    if isnothing(batchsize)
      converged = norm(dx) < reltol*norm(x) || norm(y) < abstol
    else
      # Keep going until each block in the batch converged individually
      converged = all(sum(abs2, dx; dims=1) .< reltol .* sum(abs2, x; dims=1)) || all(sum(abs2, y; dims=1) .< abstol)
    end
    if converged
      if _checkstable
        eg = eigen(jac)
        stable = all(t->norm(t)<=1, eg.values)
        return (;u=x, converged=true, n_iters=iter, stable=stable)
      else
        return (;u=x, converged=true, n_iters=iter)
      end
    end
  end
  if _checkstable
      return (;u=x, converged=false, n_iters=maxiter, stable=false)
  else
      return (;u=x, converged=false, n_iters=maxiter)
  end
end