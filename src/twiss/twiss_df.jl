function _twiss_df(colnames, twi, ::Val{as_taylor_series}) where {as_taylor_series}
  cols, colunits = _twiss_map(colnames)
  ncols = length(cols)
  nrows = length(twi.s)
  row1 = Vector{Any}(undef, ncols)

  map_cache = build_cache(typeof(twi.fac[1].a))
  tps_cache = build_cache(typeof(first(twi.fac[1].a.v)))
  float_cache = build_cache(Float64)
  smatrix4_cache = build_cache(SMatrix{4,4,Float64})
  smatrix6_cache = build_cache(SMatrix{6,6,Float64})
  vf_cache = build_cache(complex(typeof(zero(VectorField, twi.fac[1].a))))

  persistent_map_cache = build_cache(typeof(twi.fac[1].a))
  persistent_cmap_cache = build_cache(complex(typeof(twi.fac[1].a)))

  cache = TwissCache(map_cache, tps_cache, float_cache, smatrix4_cache, smatrix6_cache, vf_cache, persistent_map_cache, persistent_cmap_cache)
  for i in 1:ncols
    col = cols[i]
    row1[i] = @noinline col(1, twi, cache, Val{as_taylor_series}())
  end
  
  # Now construct DataFrame
  df = DataFrame([Vector{typeof(row1[i])}(undef, nrows) for i in 1:ncols], colnames)

  # Add unit information
  for i in 1:ncols
    colmetadata!(df, colnames[i], "unit", colunits[i]; style=:note)
  end
  df[1,:] = row1

  # enter type sdf loop
  return _twiss_df_loop(df, cols, twi, cache, Val{as_taylor_series}())
end

function _twiss_df_loop(df, cols, twi, cache, ::Val{as_taylor_series}) where {as_taylor_series}
  nrows, ncols = size(df)
  for row in 2:nrows
    for col in 1:ncols
      df[row,col] = @noinline (cols[col])(row, twi, cache, Val{as_taylor_series}())
      empty!(cache.map)
      empty!(cache.tps)
      empty!(cache.float)
      empty!(cache.smatrix4)
      empty!(cache.smatrix6)
      empty!(cache.vf)
    end
  end
  return df
end

