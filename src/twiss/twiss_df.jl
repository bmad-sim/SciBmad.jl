function _twiss_df(colnames, twi, include_start, include_end, ::Val{as_taylor_series}) where {as_taylor_series}
  cols, colunits = _twiss_map(colnames, twi)
  ncols = length(cols)
  nrows = length(twi.s)
  if !include_start
    nrows -= 1
  end
  if !include_end
    nrows -= 1
  end
  row1 = Vector{Any}(undef, ncols)

  map_cache = build_cache(typeof(twi.fac[1].a))
  tps_cache = build_cache(typeof(first(twi.fac[1].a.v)))
  float_cache = build_cache(Float64)
  matrix_cache = build_cache(Matrix{Float64})
  vf_cache = build_cache(complex(typeof(zero(VectorField, twi.fac[1].a))))

  persistent_matrix_cache = build_cache(Matrix{Float64})
  persistent_map_cache = build_cache(typeof(twi.fac[1].a))
  persistent_cmap_cache = build_cache(complex(typeof(twi.fac[1].a)))

  cache = TwissCache(map_cache, tps_cache, float_cache, matrix_cache, vf_cache, persistent_matrix_cache, persistent_map_cache, persistent_cmap_cache)
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
  if include_start
    df[1,:] = row1
    shift = 0
  else
    shift = -1
  end

  # enter type sdf loop
  return _twiss_df_loop(df, cols, twi, cache, shift, Val{as_taylor_series}()), cache
end

function _twiss_df_loop(df, cols, twi, cache, shift, ::Val{as_taylor_series}) where {as_taylor_series}
  nrows, ncols = size(df)
  for row in 2:(nrows-shift)
    for col in 1:ncols
      df[row+shift,col] = @noinline (cols[col])(row, twi, cache, Val{as_taylor_series}())
      empty!(cache.map)
      empty!(cache.tps)
      empty!(cache.float)
      empty!(cache.matrix)
      empty!(cache.vf)
    end
  end
  return df
end

