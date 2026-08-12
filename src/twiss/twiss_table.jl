function _twiss_table(colnames, twi)
  cols, collabels = _twiss_map(colnames)
  ncols = length(cols)
  nrows = length(twi.s)
  row1 = Vector{Any}(undef, ncols)

  map_cache = build_cache(typeof(twi.fac[1].a))
  tps_cache = build_cache(typeof(first(twi.fac[1].a.v)))
  float_cache = build_cache(Float64)
  smatrix4_cache = build_cache(SMatrix{4,4,Float64})
  smatrix6_cache = build_cache(SMatrix{6,6,Float64})

  persistent_map_cache = build_cache(typeof(twi.fac[1].a))

  cache = TwissCache(map_cache, tps_cache, float_cache, smatrix4_cache, smatrix6_cache, persistent_map_cache)
  for i in 1:ncols
    col = cols[i]
    row1[i] = @noinline col(1, twi, cache, Val{false}())
  end
  
  # Now construct DataFrame
  table = DataFrame([Vector{typeof(row1[i])}(undef, nrows) for i in 1:ncols], colnames)
  for i in 1:ncols
    colmetadata!(table, colnames[i], "label", collabels[i]; style=:note)
  end
  table[1,:] = row1

  # enter type stable loop
  return _twiss_table_loop(table, cols, twi, cache)
end

function _twiss_table_loop(table, cols, twi, cache)
  nrows, ncols = size(table)
  for row in 2:nrows
    for col in 1:ncols
      table[row,col] = @noinline (cols[col])(row, twi, cache, Val{false}())
      empty!(cache.map)
      empty!(cache.tps)
      empty!(cache.float)
      empty!(cache.smatrix4)
      empty!(cache.smatrix6)
    end
  end
  return table
end

