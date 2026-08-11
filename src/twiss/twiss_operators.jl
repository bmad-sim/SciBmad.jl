throwunreachable() = error("Unreachable error hit, please submit a minimal working example")

#NOTE: the H's and B's essentially define the structure of the lattice functions (LFs)
# i.e. if cache.smatrix6 has the H matrix, then all downstream LFs can safely 
# assume that there is no coasting AND no parameter dependence.
# if smatrix4 has H matrix, now coasting and no parameter dependence
# if map, then parameter dependence
# The H matrices are the first things that are computed. I assume it is safe 
# for other LFs to derive this information given which cache the H's appear in
@inline function _Hk(j, twi, cache, ::Val{as_tps}, k) where {as_tps}
  if k == 1
    sym = :H1
  elseif k == 2
    sym = :H2
  elseif k == 3
    sym = :H3
  else
    error("Index for de Moivre H matrix must be between 1 and 3")
  end

  if haskey(cache.smatrix4, sym)
    return cache.smatrix4[sym]
  elseif haskey(cache.smatrix6, sym)
    return cache.smatrix6[sym]
  elseif haskey(cache.map, sym)
    return as_tps ? cache.map[sym] : NNF.jacobian(cache.map[sym], NNF.HVARS)
  end
  
  mo = maxord(twi)
  nn = ndiffs(twi)
  coast = iscoasting(twi)
  nhv = nhvars(twi)

  if k == 3 && coast
    error("Unable to compute de Moivre matrix H3: beam is coasting")
  end

  if mo == 1 || (nn == 6 && !coast)
    if coast
      if !haskey(cache.smatrix4, :a1_mat)
        cache.smatrix4[:a1_mat] = NNF.jacobian(twi.fac[j].a1, NNF.HVARS)
        cache.smatrix4[:a1i_mat] = inv(NNF.jacobian(twi.fac[j].a1, NNF.HVARS))
      end
      a1_mat = cache.smatrix4[:a1_mat]
      a1i_mat = cache.smatrix4[:a1i_mat]
    else
      if !haskey(cache.smatrix6, :a1_mat)
        cache.smatrix6[:a1_mat] = NNF.jacobian(twi.fac[j].a1, NNF.HVARS)
        cache.smatrix6[:a1i_mat] = inv(NNF.jacobian(twi.fac[j].a1, NNF.HVARS))
      end
      a1_mat = cache.smatrix6[:a1_mat]
      a1i_mat = cache.smatrix6[:a1i_mat]
    end
    a1_matk = StaticArrays.sacollect(SMatrix{nhv,2,Float64}, a1_mat[row,col] for col in (2*k-1):(2*k) for row in 1:nhv)
    a1i_matk = StaticArrays.sacollect(SMatrix{nhv,2,Float64}, a1i_mat[row,col] for col in (2*k-1):(2*k) for row in 1:nhv)
    Hk = a1_matk * a1i_matk'
    if coast
      cache.smatrix4[sym] = Hk
    else
      cache.smatrix6[sym] = Hk
    end
    return Hk
  else
    if !haskey(cache.persistent_map, :tmp1)
      cache.persistent_map[:tmp1] = zero(twi.fac[j].a)
    end
    if !haskey(cache.map, :a1i)
      cache.map[:a1i] = inv(twi.fac[j].a1)
    end
    tmp1 = cache.persistent_map[:tmp1]
    NNF.clear!(tmp1)
    a1 = twi.fac[j].a1
    a1i = cache.map[:a1i]
    NNF.setray!(tmp1.v, v_matrix=NNF.ip_mat(a1, k))
    Hk = a1∘tmp1∘a1i
    cache.map[sym] = Hk
    return as_tps ? Hk : NNF.jacobian(Hk, NNF.HVARS)
  end
end

@inline _H1(j, twi, cache, ::Val{as_tps}) where {as_tps} = _Hk(j, twi, cache, Val{as_tps}(), 1)
@inline _H2(j, twi, cache, ::Val{as_tps}) where {as_tps} = _Hk(j, twi, cache, Val{as_tps}(), 2)
@inline _H3(j, twi, cache, ::Val{as_tps}) where {as_tps} = _Hk(j, twi, cache, Val{as_tps}(), 3)

@inline function _Bk(j, twi, cache, ::Val{as_tps}, k) where {as_tps}
  if k == 1
    sym = :B1
  elseif k == 2
    sym = :B2
  elseif k == 3
    sym = :B3
  else
    error("Index for de Moivre B matrix must be between 1 and 3")
  end

  if haskey(cache.smatrix4, sym)
    return cache.smatrix4[sym]
  elseif haskey(cache.smatrix6, sym)
    return cache.smatrix6[sym]
  elseif haskey(cache.map, sym)
    return as_tps ? cache.map[sym] : NNF.jacobian(cache.map[sym], NNF.HVARS)
  end
  
  mo = maxord(twi)
  nn = ndiffs(twi)
  coast = iscoasting(twi)
  nhv = nhvars(twi)

  if k == 3 && coast
    error("Unable to compute de Moivre matrix H3: beam is coasting")
  end

  if mo == 1 || (nn == 6 && !coast)
    if coast
      if !haskey(cache.smatrix4, :a1_mat)
        cache.smatrix4[:a1_mat] = NNF.jacobian(twi.fac[j].a1, NNF.HVARS)
        cache.smatrix4[:a1i_mat] = inv(NNF.jacobian(twi.fac[j].a1, NNF.HVARS))
      end
      a1_mat = cache.smatrix4[:a1_mat]
      a1i_mat = cache.smatrix4[:a1i_mat]
    else
      if !haskey(cache.smatrix6, :a1_mat)
        cache.smatrix6[:a1_mat] = NNF.jacobian(twi.fac[j].a1, NNF.HVARS)
        cache.smatrix6[:a1i_mat] = inv(NNF.jacobian(twi.fac[j].a1, NNF.HVARS))
      end
      a1_mat = cache.smatrix6[:a1_mat]
      a1i_mat = cache.smatrix6[:a1i_mat]
    end
    a1_matk1 = StaticArrays.sacollect(SVector{nhv,Float64}, a1_mat[row,(2*k-1)] for row in 1:nhv)
    a1_matk2 = StaticArrays.sacollect(SVector{nhv,Float64}, a1_mat[row,(2*k)] for row in 1:nhv)
    a1i_matk1 = StaticArrays.sacollect(SVector{nhv,Float64}, a1i_mat[row,(2*k-1)] for row in 1:nhv)
    a1i_matk2 = StaticArrays.sacollect(SVector{nhv,Float64}, a1i_mat[row,(2*k)] for row in 1:nhv)
    Bk = a1_matk1*transpose(a1i_matk2) - a1_matk2*transpose(a1i_matk1)
    if coast
      cache.smatrix4[sym] = Bk
    else
      cache.smatrix6[sym] = Bk
    end
    return Bk
  else
    if !haskey(cache.persistent_map, :tmp1)
      cache.persistent_map[:tmp1] = zero(twi.fac[j].a)
    end
    if !haskey(cache.map, :a1i)
      cache.map[:a1i] = inv(twi.fac[j].a1)
    end
    tmp1 = cache.persistent_map[:tmp1]
    NNF.clear!(tmp1)
    a1 = twi.fac[j].a1
    a1i = cache.map[:a1i]
    NNF.setray!(tmp1.v, v_matrix=NNF.jp_mat(a1, k))
    Bk = a1∘tmp1∘a1i
    cache.map[sym] = Bk
    return as_tps ? Bk : NNF.jacobian(Bk, NNF.HVARS)
  end
end

@inline _B1(j, twi, cache, ::Val{as_tps}) where {as_tps} = _Bk(j, twi, cache, Val{as_tps}(), 1)
@inline _B2(j, twi, cache, ::Val{as_tps}) where {as_tps} = _Bk(j, twi, cache, Val{as_tps}(), 2)
@inline _B3(j, twi, cache, ::Val{as_tps}) where {as_tps} = _Bk(j, twi, cache, Val{as_tps}(), 3)


@inline function _gammac(j, twi, cache, ::Val{as_tps}) where {as_tps}
  if haskey(cache.float, :gammac)
    return cache.float[:gammac]
  elseif haskey(cache.tps, :gammac)
    return as_tps ? cache.tps[:gammac] : scalar(cache.tps[:gammac])
  elseif !haskey(cache.map, :H1) && !(haskey(cache.smatrix4, :H1) || haskey(cache.smatrix6, :H1))
    _H1(j, twi, cache, Val{as_tps}())
  end

  if haskey(cache.smatrix4, :H1)
    gammac = sqrt(cache.smatrix4[:H1][1,1])
    cache.float[:gammac] = gammac
    return gammac
  elseif haskey(cache.smatrix6, :H1)
    gammac = sqrt(cache.smatrix6[:H1][1,1])
    cache.float[:gammac] = gammac
    return gammac
  elseif haskey(cache.map, :H1)
    H1 = cache.map[:H1]
    mo = maxord(twi)
    gammac = TI.cutord(sqrt(NNF.factor_out(H1.v[1], 1)), mo)
    cache.tps[:gammac] = gammac
    return as_tps ? gammac : scalar(gammac)
  end
end

@inline function _ckl(j, twi, cache, ::Val{as_tps}, kl) where {as_tps}
  if kl == 11
    sym = :c11
  elseif kl == 12
    sym = :c12
  elseif kl == 21
    sym = :c21
  elseif kl == 22
    sym = :c22
  else
    error("Coupling matrix index may only be 11, 12, 21, or 22")
  end

  if haskey(cache.float, sym)
    return cache.float[sym]
  elseif haskey(cache.tps, sym)
    return as_tps ? cache.tps[sym] : scalar(cache.tps[sym])
  elseif !haskey(cache.float, :gammac) && !haskey(cache.tps, :gammac)
    _gammac(j, twi, cache, Val{as_tps}())
  end

  if haskey(cache.float, :gammac)
    gammac = cache.float[:gammac]
    if !haskey(cache.smatrix4, :H1)
      H1_6 = cache.smatrix6[:H1]
      cache.smatrix4[:H1] = StaticArrays.sacollect(SMatrix{4,4,Float64}, H1_6[row,col] for col in 1:4 for row in 1:4)
    end
    H1 = cache.smatrix4[:H1]
    c11 = -H1[1,3]/gammac
    c12 = -H1[1,4]/gammac
    c21 = -H1[2,3]/gammac
    c22 = -H1[2,4]/gammac
    cache.float[:c11] = c11
    cache.float[:c21] = c21
    cache.float[:c12] = c12
    cache.float[:c22] = c22
    return cache.float[sym]
  elseif haskey(cache.tps, :gammac)
    gammac = cache.tps[:gammac]
    H1 = cache.map[:H1]
    mo = maxord(twi)

    c11 = TI.cutord(-NNF.factor_out(H1.v[1], 3)/gammac, mo)
    c12 = TI.cutord(-NNF.factor_out(H1.v[1], 4)/gammac, mo)
    c21 = TI.cutord(-NNF.factor_out(H1.v[2], 3)/gammac, mo)
    c22 = TI.cutord(-NNF.factor_out(H1.v[2], 4)/gammac, mo)
    cache.tps[:c11] = c11
    cache.tps[:c21] = c21
    cache.tps[:c12] = c12
    cache.tps[:c22] = c22
    return as_tps ? cache.tps[sym] : scalar(cache.tps[sym])
  else
    throwunreachable()
  end
end

@inline _c11(j, twi, cache, ::Val{as_tps}) where {as_tps} = _ckl(j, twi, cache, Val{as_tps}(), 11)
@inline _c12(j, twi, cache, ::Val{as_tps}) where {as_tps} = _ckl(j, twi, cache, Val{as_tps}(), 12)
@inline _c21(j, twi, cache, ::Val{as_tps}) where {as_tps} = _ckl(j, twi, cache, Val{as_tps}(), 21)
@inline _c22(j, twi, cache, ::Val{as_tps}) where {as_tps} = _ckl(j, twi, cache, Val{as_tps}(), 22)

@inline function _Vi(j, twi, cache, ::Val{as_tps}) where {as_tps}
  if haskey(cache.smatrix4, :Vi)
    return cache.smatrix4[:Vi]
  elseif haskey(cache.map, :Vi)
    Vi = cache.map[:Vi]
    return as_tps ? Vi : StaticArrays.sacollect(SMatrix{4,4,Float64}, TI.geti(Vi.v[row], col) for col in 1:4 for row in 1:4)
  elseif !haskey(cache.float, :c11) && !haskey(cache.tps, :c11)
    _c11(j, twi, cache, Val{as_tps}()) # all coupling matrix components computed when this is executed
  end

  if haskey(cache.float, :c11)
    c11 = cache.float[:c11]
    c12 = cache.float[:c12]
    c21 = cache.float[:c21]
    c22 = cache.float[:c22]
    gammac = cache.float[:gammac]

    C = SA[c11 c12; c21 c22]
    Ct = SA[c22 -c12; -c21 c11]
    Vi = gammac*I + vcat(hcat(zero(C), -C), hcat(Ct, zero(Ct)))
    cache.smatrix4[:Vi] = Vi
    return Vi
  elseif haskey(cache.tps, :c11)
    c11 = cache.tps[:c11]
    c12 = cache.tps[:c12]
    c21 = cache.tps[:c21]
    c22 = cache.tps[:c22]
    gammac = cache.tps[:gammac]

    C = SA[c11 c12; c21 c22]
    Ct = SA[c22 -c12; -c21 c11]
    Vi_mat = gammac*I + vcat(hcat(zero.(C), -C), hcat(Ct, zero.(Ct)))
    Vi = zero(twi.fac[j].a)
    if iscoasting(twi)
      TI.seti!(Vi.v[6], 1, 6) # identity in delta if coasting 
    end
    for row in 1:4
      for col in 1:4
        NNF.factor_in!(Vi.v[row], Vi_mat[row,col], col) # this should work + be faster
        #TI.add!(Vi.v[row], Vi.v[row], NNF.factor_in(Vi_mat[row,col], col))
      end
    end
    cache.map[:Vi] = Vi
    return as_tps ? Vi : StaticArrays.sacollect(SMatrix{4,4,Float64}, TI.geti(Vi.v[row], col) for col in 1:4 for row in 1:4)
  else
    throwunreachable()
  end
end

@inline function _N(j, twi, cache, ::Val{as_tps}) where {as_tps}
  if haskey(cache.smatrix4, :N)
    return cache.smatrix4[:N]
  elseif haskey(cache.map, :N)
    N = cache.map[:N]
    return as_tps ? N : StaticArrays.sacollect(SMatrix{4,4,Float64}, TI.geti(N.v[row], col) for col in 1:4 for row in 1:4)
  elseif !haskey(cache.smatrix4, :Vi) && !haskey(cache.map, :Vi)
    _Vi(j, twi, cache, Val{as_tps}()) # forces computation of Vi
  end

  if haskey(cache.smatrix4, :Vi)
    if !haskey(cache.smatrix4, :a1_mat)
      a1_mat6 = cache.smatrix6[:a1_mat]
      cache.smatrix4[:a1_mat] = StaticArrays.sacollect(SMatrix{4,4,Float64}, a1_mat6[row,col] for col in 1:4 for row in 1:4)
    end
    Vi = cache.smatrix4[:Vi]
    a1_mat = cache.smatrix4[:a1_mat]
    N = Vi*a1_mat
    cache.smatrix4[:N] = N
    return N
  elseif haskey(cache.map, :Vi)
    N = cache.map[:Vi] ∘ twi.fac[j].a1 
    cache.map[:N] = N
    return as_tps ? N : StaticArrays.sacollect(SMatrix{4,4,Float64}, TI.geti(N.v[row], col) for col in 1:4 for row in 1:4)
  else
    throwunreachable()
  end
end

@inline function _betak(j, twi::TwissInternal, cache::TwissCache, ::Val{as_tps}, k) where {as_tps} 
  if k == 1
    sym = :beta1
  elseif k == 2
    sym = :beta2
  else
    error("Only 4D Teng-Edwards beta functions are currently supported. Use de Moivre instead")
  end

  if haskey(cache.float, sym) 
    return cache.float[sym]
  elseif haskey(cache.tps, sym)
    return as_tps ? cache.tps[sym] : scalar(cache.tps[sym])
  elseif !haskey(cache.smatrix4, :N) && !haskey(cache.map, :N)
    _N(j, twi, cache, Val{as_tps}()) # Forces computation of N and storage in cache
  end

  # Now N will exist in one of the two:
  if haskey(cache.smatrix4, :N)
    N = cache.smatrix4[:N]
    betak = N[2*k-1,2*k-1]^2
    cache.float[sym] = betak
    return betak
  elseif haskey(cache.map, :N)
    N = cache.map[:N]
    mo = maxord(twi)
    betak = TI.cutord(NNF.factor_out(N.v[2*k-1], 2*k-1)^2, mo)
    cache.tps[sym] = betak
    return as_tps ? betak : scalar(betak)
  else
    throwunreachable()
  end
end

@inline _beta1(j, twi, cache, ::Val{as_tps}) where {as_tps} = _betak(j, twi, cache, Val{as_tps}(), 1)
@inline _beta2(j, twi, cache, ::Val{as_tps}) where {as_tps} = _betak(j, twi, cache, Val{as_tps}(), 2)

@inline function _alphak(j, twi::TwissInternal, cache::TwissCache, ::Val{as_tps}, k) where {as_tps}
  if k == 1
    sym = :alpha1
  elseif k == 2
    sym = :alpha2
  else
    error("Only 4D Teng-Edwards alpha functions are currently supported. Use de Moivre instead")
  end

  if haskey(cache.float, sym) 
    return cache.float[sym]
  elseif haskey(cache.tps, sym)
    return as_tps ? cache.tps[sym] : scalar(cache.tps[sym])
  elseif !haskey(cache.smatrix4, :N) && !haskey(cache.map, :N)
    _N(j, twi, cache, Val{as_tps}()) # Forces computation of N and storage in cache
  end

  # Now N will exist in one of the two:
  if haskey(cache.smatrix4, :N)
    N = cache.smatrix4[:N]
    alphak = -N[2*k,2*k-1]*N[2*k-1,2*k-1]
    cache.float[sym] = alphak
    return alphak
  elseif haskey(cache.map, :N)
    N = cache.map[:N]
    mo = maxord(twi)
    alphak = -TI.cutord(NNF.factor_out(N.v[2*k], 2*k-1)*NNF.factor_out(N.v[2*k-1], 2*k-1), mo)
    cache.tps[sym] = alphak
    return as_tps ? alphak : scalar(alphak)
  else
    throwunreachable()
  end
end

@inline _alpha1(j, twi, cache, ::Val{as_tps}) where {as_tps} = _alphak(j, twi, cache, Val{as_tps}(), 1)
@inline _alpha2(j, twi, cache, ::Val{as_tps}) where {as_tps} = _alphak(j, twi, cache, Val{as_tps}(), 2)

# orbit, like the H matrix, also is a "fundamental" quantity where what is stored 
# in the cache defines certain features (delta-dependent/parameter dependent)
@inline function _orbit(j, twi, cache, ::Val{as_tps}, k) where {as_tps}
  if k == 1
    sym = :x
  elseif k == 2
    sym = :px
  elseif k == 3
    sym = :y
  elseif k == 4
    sym = :py
  elseif k == 5
    sym = :z
  elseif k == 6
    sym = :pz
  else
    error("Orbit index must be between 1 and 6")
  end

  if haskey(cache.float, sym)
    return cache.float[sym]
  elseif haskey(cache.tps, sym)
    return as_tps ? cache.tps[sym] : scalar(cache.tps)
  end

  o = twi.fac[j].a0.v[k]
  nn = ndiffs(twi)

  # Check what's going on with the orbit - should we store as TPS or float?
  if (iscoasting(twi) && k < 5) || nn > 6 # then cache as tps
    vk = zero(o)
    TI.copy_tps!(vk, o)
    if k < 6 || !iscoasting(twi)
      TI.seti!(vk, 0, k)
    end
    cache.tps[sym] = vk
    return as_tps ? vk : scalar(vk)
  else
    cache.float[sym] = scalar(o)
    return scalar(o)
  end
end

@inline _x( j, twi, cache, ::Val{as_tps}) where {as_tps}  = _orbit(j, twi, cache, Val{as_tps}(), 1)
@inline _px(j, twi, cache, ::Val{as_tps}) where {as_tps}  = _orbit(j, twi, cache, Val{as_tps}(), 2)
@inline _y( j, twi, cache, ::Val{as_tps}) where {as_tps}  = _orbit(j, twi, cache, Val{as_tps}(), 3)
@inline _py(j, twi, cache, ::Val{as_tps}) where {as_tps}  = _orbit(j, twi, cache, Val{as_tps}(), 4)
@inline _z( j, twi, cache, ::Val{as_tps}) where {as_tps}  = _orbit(j, twi, cache, Val{as_tps}(), 5)
@inline _pz(j, twi, cache, ::Val{as_tps}) where {as_tps}  = _orbit(j, twi, cache, Val{as_tps}(), 6)


@inline function _phi(j, twi, cache, ::Val{as_tps}, k) where {as_tps}
  if k == 1
    sym = :phi1
  elseif k == 2
    sym = :phi2
  elseif k == 3
    sym = :phi3_or_slip
  else
    error("Phase advance index must be between 1 and 3")
  end
  phi = getproperty(twi, sym)[j]

  # Note that if canonise=0, then slip will just be a scalar
  # else, slip will be a TPSA in delta/parameters, and the appropriate monomial (first order term)
  # must be extracted depending on what people want

  # If as_tps = true, then the only option is to return as a full TPS
  # if as_tps = false, then presumably no parameters, so people want 
  # linear slip in delta
  if as_tps
    return phi
  else
    if k == 3 && iscoasting(twi) && TI.is_tps_type(typeof(phi)) isa TI.IsTPSType
      return TI.geti(phi, 6)
    else
      return scalar(phi)
    end
  end
end

@inline _phi1(j, twi, cache, ::Val{as_tps}) where {as_tps} = _phi(j, twi, cache, Val{as_tps}(), 1)
@inline _phi2(j, twi, cache, ::Val{as_tps}) where {as_tps} = _phi(j, twi, cache, Val{as_tps}(), 2)
@inline function _phi3(j, twi, cache, ::Val{as_tps}) where {as_tps}
  if k == 3 && iscoasting(twi) 
    error("Cannot compute phi3: beam is coasting")
  end
  return _phi(j, twi, cache, Val{as_tps}(), 3)
end

@inline function _slip(j, twi, cache, ::Val{as_tps}) where {as_tps}
  # convert z (bmad) to tau (mad)
  # Need to include parameter/delta dependence in Lorentz beta...
  beta_gamma_ref = twi.beta_gamma_ref[j]
  tilde_m = 1/beta_gamma_ref
  if !haskey(cache.float, :pz) && !haskey(cache.tps, :pz)
    _pz(j, twi, cache, Val{true}()) # Force (potentially TPSA) calculation of z
  end

  if haskey(cache.tps, :pz)
    pz = cache.tps[:pz]
  elseif haskey(cache.float, :pz)
    pz = cache.float[:pz]
  else
    throwunreachable()
  end

  rel_p = 1 + pz
  beta = rel_p/sqrt(rel_p*rel_p + tilde_m*tilde_m)

  phi3 = _phi(j, twi, cache, Val{as_tps}(), 3)

  if iscoasting(twi)
    return phi3 / beta
  end 

  # Else use the approximation of Etienne
  # B3[5,6]*sin(phi[3]*2*pi)
  if !haskey(cache.smatrix6, :B3) && !haskey(cache.map, :B3)
    _B3(j, twi, cache, Val{as_tps}())
  end
  if haskey(cache.smatrix6, :B3) # Then no parameter dependence
    return cache.smatrix6[:B3][5,6]*sin(2*pi*phi3) / beta
  elseif haskey(cache.map, :B3) # Parameter dependence
    B3 = cache.map[:B3]
    slip = NNF.factor_out(B3.v[5], 6) / beta
    return as_tps ? slip : scalar(slip)
  else
    throwunreachable()
  end
end

@inline function _linear_dispersion(j, twi, cache, ::Val{as_tps}, k) where {as_tps}
  if k == 1
    sym = :dx
    osym = :x
  elseif k == 2
    sym = :dpx
    osym = :px
  elseif k == 3
    sym = :dy
    osym = :y
  elseif k == 4
    sym = :dpy
    osym = :py
  else
    error("Linear dispersion index must be between 1 and 4")
  end

  if haskey(cache.smatrix6, :H3) # Then no parameter dependence + not coasting
    return cache.smatrix6[:H3][k,6]
  elseif haskey(cache.tps, sym) # Potential parameter dependence and/or coasting
    return as_tps ? cache.tps[sym] : scalar(cache.tps[sym])
  end

  if iscoasting(twi)
    if !haskey(cache.tps, osym)
      _orbit(j, twi, cache, Val{as_tps}(), k) # Force calculation of taylor series
    end
    # Get the delta-dependent part:
    dk = TI.deriv(cache.tps[osym], 6)
    cache.tps[sym] = dk
    return as_tps ? dk : scalar(dk)
  end
  
  # Approximation described by Etienne
  if !haskey(cache.smatrix6, :H3) && !haskey(cache.map, :H3)
    _H3(j, twi, cache, Val{as_tps}())
  end
  if haskey(cache.smatrix6, :H3) # Then no parameter dependence, don't store in cache
    return cache.smatrix6[:H3][k,6]
  elseif haskey(cache.map, :H3) # Parameter dependence, store in cache.tps
    H3 = cache.map[:H3]
    dk = NNF.factor_out(H3.v[k], 6)
    cache.tps[sym] = dk
    return as_tps ? dk : scalar(dk)
  else
    throwunreachable()
  end
end

@inline _dx( j, twi, cache, ::Val{as_tps}) where {as_tps} = _linear_dispersion(j, twi, cache, Val{as_tps}(), 1)
@inline _dpx(j, twi, cache, ::Val{as_tps}) where {as_tps} = _linear_dispersion(j, twi, cache, Val{as_tps}(), 2)
@inline _dy( j, twi, cache, ::Val{as_tps}) where {as_tps} = _linear_dispersion(j, twi, cache, Val{as_tps}(), 3)
@inline _dpy(j, twi, cache, ::Val{as_tps}) where {as_tps} = _linear_dispersion(j, twi, cache, Val{as_tps}(), 4)

@inline function _zeta(j, twi, cache, ::Val{as_tps}, k) where {as_tps}
  if k == 1
    sym = :zx
  elseif k == 2
    sym = :zpx
  elseif k == 3
    sym = :zy
  elseif k == 4
    sym = :zpy
  else
    error("zeta (AKA crab dispersion) index must be between 1 and 4")
  end

  if iscoasting(twi)
    error("Unable to compute zeta (AKA crab dispersion) in ring with coasting beam")
  end

  if haskey(cache.smatrix6, :H3) # no parameter dependence
    return cache.smatrix6[:H3][k,5]
  elseif haskey(cache.tps, sym)
    return as_tps ? cache.tps[sym] : scalar(cache.tps[sym])
  end

  # Approximation described by Etienne
  if !haskey(cache.smatrix6, :H3) && !haskey(cache.map, :H3)
    _H3(j, twi, cache, Val{as_tps}())
  end
  if haskey(cache.smatrix6, :H3) # Then no parameter dependence, don't store in cache
    return cache.smatrix6[:H3][k,5]
  elseif haskey(cache.map, :H3) # Parameter dependence, store in cache.tps
    H3 = cache.map[:H3]
    zk = NNF.factor_out(H3.v[k], 5)
    cache.tps[sym] = zk
    return as_tps ? zk : scalar(zk)
  else
    throwunreachable()
  end
end

@inline _zx( j, twi, cache, ::Val{as_tps}) where {as_tps} = _zeta(j, twi, cache, Val{as_tps}(), 1)
@inline _zpx(j, twi, cache, ::Val{as_tps}) where {as_tps} = _zeta(j, twi, cache, Val{as_tps}(), 2)
@inline _zy( j, twi, cache, ::Val{as_tps}) where {as_tps} = _zeta(j, twi, cache, Val{as_tps}(), 3)
@inline _zpy(j, twi, cache, ::Val{as_tps}) where {as_tps} = _zeta(j, twi, cache, Val{as_tps}(), 4)



#=
@inline function _nonlin_dispersion(j, twi, cache, ::Val{as_tps}, k, order) where {as_tps}
  if !iscoasting(twi)
    error("Higher-order dispersion can only be calculated with coasting beam (no longitudinal oscillations)")
  end

end
=#