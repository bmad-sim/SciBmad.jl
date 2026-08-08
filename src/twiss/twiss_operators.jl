throwunreachable() = error("Unreachable error hit, please submit a minimal working example")

@inline function _H1(j, twi, cache)
  if haskey(cache.smatrix4, :H1)
    return cache.smatrix4[:H1]
  elseif haskey(cache.smatrix6, :H1)
    return cache.smatrix6[:H1]
  elseif haskey(cache.map, :H1)
    return NNF.jacobian(cache.map[:H1], NNF.HVARS)
  end
  
  mo = maxord(twi)
  nn = ndiffs(twi)
  coast = iscoasting(twi)
  nhv = nhvars(twi)

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
    a1_mat1 = StaticArrays.sacollect(SMatrix{nhv,2,Float64}, a1_mat[row,col] for col in 1:2 for row in 1:nhv)
    a1i_mat1 = StaticArrays.sacollect(SMatrix{nhv,2,Float64}, a1i_mat[row,col] for col in 1:2 for row in 1:nhv)
    a1_mat2 = StaticArrays.sacollect(SMatrix{nhv,2,Float64}, a1_mat[row,col] for col in 3:4 for row in 1:nhv)
    a1i_mat2 = StaticArrays.sacollect(SMatrix{nhv,2,Float64}, a1i_mat[row,col] for col in 3:4 for row in 1:nhv)
    H1 = a1_mat1 * a1i_mat1'
    H2 = a1_mat2 * a1i_mat2'
    if coast
      cache.smatrix4[:H1] = H1
      cache.smatrix4[:H2] = H2
    else
      a1_mat2 = StaticArrays.sacollect(SMatrix{nhv,2,Float64}, a1_mat[row,col] for col in 5:6 for row in 1:nhv)
      a1i_mat2 = StaticArrays.sacollect(SMatrix{nhv,2,Float64}, a1i_mat[row,col] for col in 5:6 for row in 1:nhv)
      H3 = a1_mat3 * a1i_mat3'
      cache.smatrix6[:H1] = H1
      cache.smatrix6[:H2] = H2
      cache.smatrix6[:H3] = H3
    end
    return H1
  else
    tmp = zero(twi.fac.a)
    a1 = twi.fac.a1
    a1i = inv(a1)
    H = StaticArrays.sacollect(SVector{div(nhv, 2)}, begin
        setray!(tmp.v, v_matrix=NNF.ip_mat(a1, i))
        a1∘tmp∘a1i
      end for i in 1:div(nhv, 2)
    )
    cache.map[:H1] = H[1]
    cache.map[:H2] = H[2]
    if coast
      cache.map[:H3] = H[3]
    end
    return NNF.jacobian(H[1], NNF.HVARS)
  end
end

@inline function _H2(j, twi, cache) 
  if !haskey(cache.smatrix4, :H2) && !haskey(cache.smatrix6, :H2) && !haskey(cache.map, :H2)
    _H1(j, twi, cache)
  end
  if haskey(cache.smatrix4, :H2)
    return cache.smatrix4[:H2]
  elseif haskey(cache.smatrix6, :H2)
    return cache.smatrix6[:H2]
  elseif haskey(cache.map, :H2)
    return cache.map[:H2]
  else
    throwunreachable()
  end
end

@inline function _H3(j, twi, cache) 
  if iscoasting(twi)
    error("Cannot compute de Moivre matrix H3: no synchrotron oscillations")
  end
  if !haskey(cache.smatrix4, :H3) && !haskey(cache.smatrix6, :H3) && !haskey(cache.map, :H3)
    _H1(j, twi, cache)
  end
  if haskey(cache.smatrix4, :H3)
    return cache.smatrix4[:H3]
  elseif haskey(cache.smatrix6, :H3)
    return cache.smatrix6[:H3]
  elseif haskey(cache.map, :H3)
    return cache.map[:H3]
  else
    throwunreachable()
  end
end

@inline function _gammac(j, twi, cache)
  if haskey(cache.float, :gammac)
    return cache.float[:gammac]
  elseif haskey(cache.tps, :gammac)
    return scalar(cache.tps[:gammac])
  elseif !haskey(cache.map, :H1) && !(haskey(cache.smatrix4, :H1) || haskey(cache.smatrix6, :H1))
    _H1(j, twi, cache)
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
    gammac = sqrt(cache.smatrix4[:H1][1,1])
    cache.float[:gammac] = gammac
    return gammac
  end
end

@inline function _c11(j, twi, cache)
  if haskey(cache.float, :c11)
    return cache.float[:c11]
  elseif haskey(cache.tps, :c11)
    return scalar(cache.tps[:c11])
  elseif !haskey(cache.float, :gammac) && !haskey(cache.tps, :gammac)
    _gammac(j, twi, cache)
  end
  
  if haskey(cache.float, :gammac)
    gammac = cache.float[:gammac]
    if !haskey(cache.smatrix4, :H1)
      H1_6 = cache.smatrix6[:H1]
      cache.smatrix4[:H1] = StaticArrays.sacollect(SMatrix{4,4,Float64}, H1_6[row,col] for col in 1:4 for row in 1:4)
    end
    H1 = cache.smatrix4[:H1]
    c11 = H1[1,1]/gammac
    c12 = H1[1,2]/gammac
    c21 = H1[2,1]/gammac
    c22 = H1[2,2]/gammac
    cache.float[:c11] = c11
    cache.float[:c21] = c21
    cache.float[:c12] = c12
    cache.float[:c22] = c22
    return scalar(c11)
  elseif haskey(cache.tps, :gammac)
    gammac = cache.tps[:gammac]
    H1 = cache.map[:H1]
    mo = maxord(twi)

    c11 = TI.cutord(-NNF.factor_out(H1.v[1], 1)/gammac, mo)
    c12 = TI.cutord(-NNF.factor_out(H1.v[1], 2)/gammac, mo)
    c21 = TI.cutord(-NNF.factor_out(H1.v[2], 1)/gammac, mo)
    c22 = TI.cutord(-NNF.factor_out(H1.v[2], 2)/gammac, mo)
    cache.tps[:c11] = c11
    cache.tps[:c21] = c21
    cache.tps[:c12] = c12
    cache.tps[:c22] = c22
    return scalar(c11)
  else
    throwunreachable()
  end
end

@inline function _c12(j, twi, cache) 
  if !haskey(cache.float, :c12) && !haskey(cache.tps, :c12)
    _c11(j, twi, cache)
  end
  if haskey(cache.float, :c12)
    return cache.float[:c12]
  elseif haskey(cache.tps, :c12)
    return scalar(cache.tps[:c12])
  else
    throwunreachable()
  end
end

@inline function _c21(j, twi, cache) 
  if !haskey(cache.float, :c21) && !haskey(cache.tps, :c21)
    _c11(j, twi, cache)
  end
  if haskey(cache.float, :c21)
    return cache.float[:c21]
  elseif haskey(cache.tps, :c21)
    return scalar(cache.tps[:c21])
  else
    throwunreachable()
  end
end

@inline function _c22(j, twi, cache) 
  if !haskey(cache.float, :c22) && !haskey(cache.tps, :c22)
    _c11(j, twi, cache)
  end
  if haskey(cache.float, :c22)
    return cache.float[:c22]
  elseif haskey(cache.tps, :c22)
    return scalar(cache.tps[:c22])
  else
    throwunreachable()
  end
end


@inline function _Vi(j, twi, cache)
  if haskey(cache.smatrix4, :Vi)
    return cache.smatrix4[:Vi]
  elseif haskey(cache.map, :Vi)
    Vi = cache.map[:Vi]
    return StaticArrays.sacollect(SMatrix{4,4,Float64}, Vi[row][col] for col in 1:4 for row in 1:4)
  elseif !haskey(cache.float, :c11) && !haskey(cache.tps, :c11)
    _c11(j, twi, cache) # all coupling matrix components computed when this is executed
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
    Vi_mat = gammac*I + vcat(hcat(zero(C), -C), hcat(Ct, zero(Ct)))
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
    return Vi_mat
  else
    throwunreachable()
  end
end

@inline function _N(j, twi, cache)
  if haskey(cache.smatrix4, :N)
    return cache.smatrix4[:N]
  elseif haskey(cache.map, :N)
    N = cache.map[:N]
    return StaticArrays.sacollect(SMatrix{4,4,Float64}, N[row][col] for col in 1:4 for row in 1:4)
  elseif !haskey(cache.smatrix4, :Vi) && !haskey(cache.map, :Vi)
    _Vi(j, twi, cache) # forces computation of Vi
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
    return StaticArrays.sacollect(SMatrix{4,4,Float64}, N.v[row][col] for col in 1:4 for row in 1:4)
  else
    throwunreachable()
  end
end

@inline function _beta_1(j, twi::TwissInternal, cache::TwissCache)
  if haskey(cache.float, :beta_1) 
    return cache.float[:beta_1]
  elseif haskey(cache.tps, :beta_1)
    return scalar(cache.tps[:beta_1])
  elseif !haskey(cache.smatrix4, :N) && !haskey(cache.map, :N)
    _N(j, twi, cache) # Forces computation of N and storage in cache
  end

  # Now N will exist in one of the two:
  if haskey(cache.smatrix4, :N)
    N = cache.smatrix4[:N]
    beta_1 = N[1,1]^2
    cache.float[:beta_1] = beta_1
    return beta_1
  elseif haskey(cache.map, :N)
    N = cache.map[:N]
    mo = maxord(twi)
    beta_1 = TI.cutord(NNF.factor_out(N.v[1], 1)^2, mo)
    cache.tps[:beta_1] = beta_1
    return scalar(beta_1)
  else
    throwunreachable()
  end
end
