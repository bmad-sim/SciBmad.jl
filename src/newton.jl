"""
    newton!(f!, y, x; reltol=1e-13, abstol=1e-13,  max_iter=100, backend=DI.AutoForwardDiff(), check_stable=Val{false}())

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
    f!::Function,  # DO NOT SPECIALIZE ON FUNCTION! This is the mistake SciML makes...
    y::Y, 
    x::X,
    p;             # Parameters
    reltol=1e-13,
    abstol=1e-13, 
    max_iter=100, 
    # On GPU need to use ForwardDiff from primitive (pushforward) for no scalar indexing
    backend=KA.get_backend(x) isa KA.GPU ? DI.AutoForwardFromPrimitive(AutoForwardDiff()) : DI.AutoForwardDiff(),
    prep=nothing, 
    check_stable::Val{S}=Val{false}(),
    lambda=1,
    dx=zero.(x), # Temporary
) where {Y,X,S}
    if isnothing(prep)
        prep = DI.prepare_jacobian(f!, y, backend, x, DI.Constant(p))
    end
    if Y <: StaticArray && X <: StaticArray
        jac = similar(y, Size(length(Y), length(X)))
    else
        jac = similar(y, length(y), length(x))
    end
    let _f! = f!, _prep = prep, _backend = backend
        val_and_jac!(_y, _jac, _x, _p) = DI.value_and_jacobian!(_f!, _y, _jac, _prep, _backend, _x, DI.Constant(_p))
        return newton!(val_and_jac!, y, jac, x, p; reltol=reltol, abstol=abstol, max_iter=max_iter, check_stable=check_stable, lambda=lambda, dx=dx)
    end
end

function newton!(
    val_and_jac!::Function,
    y,
    jac,
    x,
    p; # Parameters
    reltol=1e-9,
    abstol=1e-14, 
    max_iter=100, 
    check_stable::Val{S}=Val{false}(),
    lambda=1,
    dx=zero.(x),
) where {S}
    for iter in 1:max_iter
        val_and_jac!(y, jac, x, p)
        dx .= lambda.*(-jac \ y)
        if norm(dx) < reltol*norm(x) || norm(y) < abstol
            x .= x .+ dx
            if S
                eg = eigen(jac)
                stable = all(t->norm(t)<=1, eg.values)
                return (;u=x, converged=true, n_iters=iter, stable=stable)
            else
                return (;u=x, converged=true, n_iters=iter)
            end
        end
        x .= x .+ dx
    end
    if S
        return (;u=x, converged=false, n_iters=max_iter, stable=false)
    else
        return (;u=x, converged=false, n_iters=max_iter)
    end
end