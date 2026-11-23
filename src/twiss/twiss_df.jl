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

  map_cache = build_cache(typeof(twi.a0[1]))
  tps_cache = build_cache(typeof(first(twi.a0[1].v)))
  float_cache = build_cache(Float64)
  matrix_cache = build_cache(Matrix{Float64})
  vf_cache = build_cache(complex(typeof(zero(VectorField, twi.a0[1]))))

  persistent_matrix_cache = build_cache(Matrix{Float64})
  persistent_map_cache = build_cache(typeof(twi.a0[1]))
  persistent_cmap_cache = build_cache(complex(typeof(twi.a0[1])))

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

# The rows in blocks, column by column: one dynamic call per column and block (per cell it is a
# slow lookup, as the columns are different functions). The caches hold the intermediate results
# of one row, shared by its columns: each row of a block has its own
function _twiss_df_loop(df, cols, twi, cache, shift, ::Val{as_taylor_series}) where {as_taylor_series}
  nrows, ncols = size(df)
  caches = [TwissCache(build_cache(valtype(cache.map)), build_cache(valtype(cache.tps)),
                       build_cache(valtype(cache.float)), build_cache(valtype(cache.matrix)),
                       build_cache(valtype(cache.vf)), cache.persistent_matrix,
                       cache.persistent_map, cache.persistent_cmap) for _ in 1:64]
  for block in Iterators.partition(2:(nrows-shift), length(caches))
    for c in caches
      empty!(c.map)
      empty!(c.tps)
      empty!(c.float)
      empty!(c.matrix)
      empty!(c.vf)
    end
    for col in 1:ncols
      _twiss_df_column!(df[!,col], cols[col], twi, caches, block, shift, Val{as_taylor_series}())
    end
  end
  return df
end

function _twiss_df_column!(v, col::F, twi, caches, block, shift,
                           ::Val{as_taylor_series}) where {F,as_taylor_series}
  for (k, row) in enumerate(block)
    v[row+shift] = @noinline col(row, twi, caches[k], Val{as_taylor_series}())
  end
  return v
end

