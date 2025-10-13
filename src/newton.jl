"""
    newton!(f!, y, x; reltol=1e-13, abstol=1e-13,  max_iter=100, backend=DI.AutoForwardDiff(), check_stable=Val{false}())

Finds roots of f!(y, x) using Newton's method.

# Arguments
- `f!`: Function that mutates y in place with the residual vector
- `y`: Solution vector
- `x`: Initial guess

# Keyword arguments
- `abstol`: Convergence absolute tolerance (default: 1e-13)
- `reltol`: Convergence relative tolerance (default: 1e-13)
- `max_iter`: Maximum number of iterations (default: 100)

Returns `NamedTuple` containing newton search results.
"""
function newton!(
    f!::Function,  # DO NOT SPECIALIZE ON FUNCTION! This is the mistake SciML makes...
    y::Y, 
    x::X,
    p;             # Parameters
    reltol=1e-13,
    abstol=1e-13, 
    max_iter=100, 
    backend=DI.AutoForwardDiff(),
    prep=DI.prepare_jacobian(f!, y, backend, x, DI.Constant(p)),
    check_stable::Val{S}=Val{false}()
) where {Y,X,S}
    if Y <: StaticArray && X <: StaticArray
        jac = similar(y, Size(length(Y), length(X)))
    else
        jac = similar(y, length(y), length(x))
    end
    for iter in 1:max_iter
        DI.value_and_jacobian!(f!, y, jac, prep, backend, x, DI.Constant(p)) 
        # Check convergence
        if norm(y) < abstol
            if S
                eg = eigen(jac)
                stable = all(t->norm(t)<=1, eg.values)
                return (;u=y, converged=true, n_iters=iter, stable=stable)
            else
                return (;u=y, converged=true, n_iters=iter)
            end
        end
        # store in y
        y .= -jac \ y
        if norm(y) < reltol
            y .= x .+ y
            if S
                eg = eigen(jac)
                stable = all(t->norm(t)<=1, eg.values)
                return (;u=y, converged=true, n_iters=iter, stable=stable)
            else
                return (;u=y, converged=true, n_iters=iter)
            end
        end
        x .= x .+ y
    end
    if S
        return (;u=y, converged=false, n_iters=max_iter, stable=false)
    else
        return (;u=y, converged=false, n_iters=max_iter)
    end
end
