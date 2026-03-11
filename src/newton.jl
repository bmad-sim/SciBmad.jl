function default_solver(device, _y, _x, ::Val{false})
  _lx = length(_x)
  _ly = length(_y)
  let lx=_lx, ly=_ly
    return (dx, jac, y)->(reshape(dx, lx) .= -jac \ reshape(y, ly))
  end
end

# For batched on CPU, do each matrix individually
function default_solver(device, _y, _x, ::Val{true})
  _batchsize = size(_x, 2)
  _n_rows = size(_y, 1)
  _n_cols = size(_x, 1)
  let n_rows=_n_rows, n_cols=_n_cols, batchsize=_batchsize, stride=_n_rows*_n_cols
    return (dx, jac::SparseMatrixCSC, y)->begin
      for i in 1:batchsize
        jac_offset = (i-1)*stride
        curjac = reshape(view(jac.nzval, (jac_offset+1):(jac_offset+stride)), (n_rows, n_cols))
        if !ArrayInterface.issingular(curjac) # Only keep going if not singular
          dx_offset = (i-1)*n_cols
          y_offset = (i-1)*n_rows
          view(dx, (dx_offset+1):(dx_offset+n_cols)) .= -curjac \ view(y, (y_offset+1):(y_offset+n_rows))
        end
      end
      return dx
    end
  end
end

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
  batched::Val{_batched}=Val{false}(), # If Val{true}(), then batch processing will be done
  checkconverged::Val{_checkconverged}=batched isa Val{false} ? Val{true}() : Val{false}(), # will check and stop if convergence is reached
  solver::T=default_solver(KA.get_backend(x), y, x, batched), # We do specialize on the solver tho
  dx=zero.(x), # Temporary
) where {Y,X,_checkstable,_checkconverged,_batched,T}
  if _batched
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

    # If we are batch and the user has NOT specified an AutoSparse autodiff, set it up for them
    if !(autodiff isa AutoSparse)
      if !isnothing(prep)
        @warn "You specified batched and provided AD prep, but your AD autodiff is NOT AutoSparse, which is required for batched-Newton.
               Your prep will therefore NOT be used"
        prep = nothing
      end

      # Make it on the CPU
      batchsize = size(x, 2)
      n_rows = size(y, 1)
      n_cols = size(x, 1)
      nnz = batchsize * n_rows * n_cols
      rows = Vector{Int}(undef, nnz)
      cols = Vector{Int}(undef, nnz)
      idx = 1 
      for b in 0:batchsize-1
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
      pattern = sparse(d_rows, d_cols, d_mat, batchsize*n_rows, batchsize*n_cols)

      detector = ADTypes.KnownJacobianSparsityDetector(pattern)
      color = repeat(1:n_cols, outer=batchsize) 
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
    return newton!(val_and_jac!, y, jac, x, contexts...; reltol=reltol, abstol=abstol, maxiter=maxiter, checkstable=checkstable, checkconverged=checkconverged, batched=batched, solver=solver, dx=dx)
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
  checkconverged::Val{_checkconverged}=Val{true}(), # n_iter will only be return if _checkconverged == true
  batched::Val{_batched}=Val{false}(), 
  solver::T=default_solver(KA.get_backend(x), y, x, batched), 
  dx=zero.(x),
) where {_checkstable,_batched,_checkconverged,T}
  # Setup:
  out = (; u=x, jac=jac)
  if !_batched
    if _checkstable
      out = merge(out, (; stable=false))
    end
    if _checkconverged
      out = merge(out, (; converged=false, n_iters=0))
    end
    # Newton:
    dx .= 0
    for iter in 1:maxiter
      val_and_jac!(y, jac, x, contexts)
      if any(isinf, y) # Stop if infinite residual
        return out
      end
      solver(dx, jac, y)
      x .= x .+ dx
      if _checkconverged && (norm(dx) < reltol*norm(x) || norm(y) < abstol)
        @reset out.converged = true
        @reset out.n_iters = iter
        if _checkstable
          eg = eigen(jac)
          @reset out.stable = all(t->norm(t)<=1, eg.values)
        end
        return out
      end
    end
    if _checkconverged
      @reset out.n_iters=maxiter
    end
    return out
  else
    if _checkstable
      error("Stability checking for batched-Newton is not currently implemented") # TODO
      # This will include an array for stable with each element corresponding to an 
      # element in the batch
    end
    if _checkconverged
      error("Convergence checking for batched-Newton is not currently implemented") # TODO
      # This will include an array for n_iter and converged with each element corresponding
      # to an element in the batch
    end
    # Newton:
    dx .= 0
    for iter in 1:maxiter
      val_and_jac!(y, jac, x, contexts)

      # If an element in the batch has an infinite residual,
      # set that sub-Jacobian equal to the identity to ensure
      # matrix
      if any(isinf, y) # Stop if infinite residual
        return out
      end
      solver(dx, jac, y)
      x .= x .+ dx
    end
    return out
  end
end