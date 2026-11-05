"""
    stroboscopic_average(bl, v0; n_turns, seed=(0, 1, 0), kwargs...)

Return an N×3 estimate of the ISF at the initial N×6 coordinates `v0`:
`normalize(sum(R(0→k)' * seed for k in 0:n_turns))`.
Tracks a copy; storage and parallel backend follow `v0`. Keywords go to `track!`.
Increase `n_turns` to check convergence.
"""
function stroboscopic_average(bl, v0::AbstractMatrix{T}; n_turns::Integer,
                             seed=(0, 1, 0), use_cpu_multithreading=true, use_KA=true,
                             kwargs...) where {T<:AbstractFloat}
    n_turns > 0 || throw(ArgumentError("n_turns must be positive"))
    size(v0, 2) == 6 && size(v0, 1) > 0 ||
        throw(ArgumentError("v0 must be a nonempty N×6 matrix"))
    length(seed) == 3 || throw(ArgumentError("seed must have length 3"))
    s = Tuple(T.(seed))
    all(isfinite, s) && sum(abs2, s) > 0 ||
        throw(ArgumentError("seed must be finite and nonzero"))
    b = Bunch(v=copy(v0), spin=true, species=bl.species_ref,
              p_over_q_ref=bl.p_over_q_ref)
    n = similar(v0, size(v0, 1), 3)
    backend = KA.get_backend(v0)
    for j in 1:3
        selectdim(n, 2, j) .= s[j] # zeroth turn
    end
    for _ in 1:n_turns
        track!(b, bl; use_cpu_multithreading, use_KA, kwargs...)
        _strob_accumulate!(backend)(n, b.coords.q, s; ndrange=size(n, 1))
        KA.synchronize(backend)
    end
    _strob_normalize!(backend)(n; ndrange=size(n, 1))
    KA.synchronize(backend)
    return n
end

@kernel function _strob_accumulate!(n, q, s)
    i = @index(Global, Linear)
    @inbounds begin
        a, x, y, z = q[i,1], -q[i,2], -q[i,3], -q[i,4]
        d = a*a + x*x + y*y + z*z
        tx, ty, tz = 2*(y*s[3]-z*s[2]), 2*(z*s[1]-x*s[3]), 2*(x*s[2]-y*s[1])
        n[i,1] += s[1] + (a*tx + y*tz - z*ty)/d
        n[i,2] += s[2] + (a*ty + z*tx - x*tz)/d
        n[i,3] += s[3] + (a*tz + x*ty - y*tx)/d
    end
end

@kernel function _strob_normalize!(n)
    i = @index(Global, Linear)
    @inbounds begin
        r = sqrt(n[i,1]^2 + n[i,2]^2 + n[i,3]^2)
        for j in 1:3
            n[i,j] /= r
        end
    end
end
