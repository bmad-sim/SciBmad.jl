# Locations of a Beamline: (beamline index, step), the beginning of an element plus a number of its
# integration steps, and (-1, 0) the end of the beamline. The end of an element is the beginning of
# the next one. As an index, n+1 (n elements) is the end of the beamline, a zero-length element.
# twiss, track, their results and TwissExpr all find their locations here, and the s of every
# location is s at the beginning of its element (as Beamlines' ele.s) plus k*L/n

# Locations given directly (the rows that the expressions of a TwissExpr need)
struct _Locations
  v::Vector{Tuple{Int,Int}}
end

# Whether ele is sel or was made from it (through any number of copies)
function _is_or_inherits(ele, sel)
  while true
    ele === sel && return true
    pdict = getfield(ele, :pdict)
    haskey(pdict, InheritParams) || return false
    ele = pdict[InheritParams].parent
  end
end

# The UniversalParams of an element, with the length (and n_steps from ds_step) evaluated with
# the Beamline's context, as in tracking
function _universal(bl, ele)
  up = ele.UniversalParams::UniversalParams
  return up.L isa DefExpr ? Beamlines.deval(up, bl.context) : up
end

# Number of steps with a tracking callback: one per integrator step of each slice (tracking errors
# for elements tracked in fewer steps, e.g. a patch with n_steps > 1)
_nsteps(bl, i) = (up = _universal(bl, bl.line[i]); up.n_slices*up.n_steps)

# s at the beginning of every element and at the end of the beamline (n+1 values, as Beamlines'
# ele.s), the length, number of integration steps, name and kind of every element, and the number
# of steps before every element and in the whole beamline (s has the type of the sum of the
# lengths, e.g. a TPS for parametric lengths)
_line_steps(bl::Beamline) = _line_steps(bl, bl.line) # (a function barrier for the line)
function _line_steps(bl, line)
  n_ele = length(line)
  L = Vector{Any}(undef, n_ele)
  n = Vector{Int}(undef, n_ele)
  names = Vector{String}(undef, n_ele)
  kinds = Vector{String}(undef, n_ele)
  for (i, ele) in enumerate(line)
    up = _universal(bl, ele)
    L[i] = up.L
    n[i] = up.n_slices*up.n_steps
    names[i] = up.name
    kinds[i] = up.kind
  end
  s, L = _line_s(identity.(L)) # (with the type of the lengths, if they all have the same)
  return s, L, n, pushfirst!(cumsum(n), 0), names, kinds
end
function _line_s(L)
  ends = accumulate(+, L; init=0f0) # (s at the end of every element, as Beamlines sums them)
  T = isconcretetype(eltype(ends)) ? eltype(ends) : isempty(ends) ? Float32 : typeof(last(ends))
  s = Vector{T}(undef, length(L) + 1)
  s[1] = 0f0
  s[2:end] .= ends
  return s, convert(Vector{T}, L)
end

_step_s(s, L, n, i, k) = k == 0 ? s[i] : s[i] + k*L[i]/n[i]

# Beamline indexes of the elements selected by sel (n+1: the end of the beamline)
function _element_indexes(bl::Beamline, sel)
  line = bl.line
  if sel isa Integer
    1 <= sel <= length(line) + 1 ||
      error("Beamline index $sel is not in 1:$(length(line)+1) ($(length(line)+1): the end)")
    return Int[sel]
  elseif sel isa LatElement
    idxs = findall(ele -> _is_or_inherits(ele, sel), line)
    isempty(idxs) && error("Element $(sel.name) is not in the Beamline")
    return idxs
  elseif sel isa Union{AbstractString,Regex}
    eles = findelements(bl, sel)
    isempty(eles) && error("No element of the Beamline matches $(repr(sel))")
    return Int[ele.beamline_index for ele in eles]
  elseif sel isa AbstractVector
    return reduce(vcat, (_element_indexes(bl, x) for x in sel); init=Int[])
  end
  error("Unsupported location $(repr(sel))")
end

# The beamline index of the element `what` (start, stop), given as a beamline index or as an
# element placed once
function _element_index(bl, sel, what)
  idxs = _element_indexes(bl, sel)
  length(idxs) == 1 || error("The `$what` element must be in the beamline exactly once")
  return only(idxs)
end

# The location k integration steps after the beginning of element i (k = its number of steps: its
# end, the beginning of the next element; i = n+1: the end of the beamline)
function _location(bl, i, k)
  n_ele = length(bl.line)
  1 <= i <= n_ele + 1 || error("Beamline index $i is not in 1:$(n_ele+1) ($(n_ele+1): the end)")
  k == 0 && return i == n_ele + 1 ? (-1, 0) : (Int(i), 0)
  if i == n_ele + 1
    0 <= k <= 1 || error("There are no integration steps after the end of the beamline")
    return (-1, 0)
  end
  n = _nsteps(bl, i)
  0 <= k <= n || error("Element $i has $n integration steps: step $k does not exist")
  return k < n ? (i, k) : i == n_ele ? (-1, 0) : (i + 1, 0)
end

# Whether location l is in the part of the beamline from the beginning of element i1 to the
# beginning of element i2 (n+1: the end of the beamline), both included
function _in_part(bl, (i, k), (i1, i2))
  i == -1 && return i2 == length(bl.line) + 1
  return i1 <= i < i2 || (i == i2 && k == 0)
end

_location_key((i, k)) = (i == -1 ? typemax(Int) : i, k)

# The locations of a column indexed with args in the part of the beamline from the beginning of
# element part[1] to the beginning of part[2], and whether they give one value (else a vector),
# for Twiss tables, TwissExpr and TrackResult alike. A beamline index or an element (with a
# position other than `:`) gives one value: the index must be in the part, and the element must
# have exactly one placement there (else an error: `[[ele]]` is every placement). Names, vectors,
# s-ranges and `:` give their locations in the part (a vector). With strict=false (for `at`), an
# index outside the part is left out and an element gives all its placements in the part.
# (ls: _line_steps(bl) if already computed)
function _column_locations(bl, part, sel; strict=true, ls=nothing)
  sel isa Integer && return _single_location(bl, part, _location(bl, sel, 0), strict), true
  if sel isa Union{Colon,Tuple{Real,Real}}
    range = sel isa Colon ? nothing : sel
    return _range_locations(bl, part, range, isnothing(ls) ? _line_steps(bl) : ls), false
  end
  idxs = _element_indexes(bl, sel)
  if sel isa LatElement && strict
    return _one_placement(bl, part, sel, idxs, [[_location(bl, i, 0)] for i in idxs]), true
  end
  locs = [_location(bl, i, 0) for i in idxs]
  return filter!(l -> _in_part(bl, l, part), locs), false
end
function _column_locations(bl, part, sel, pos; strict=true, ls=nothing)
  idxs = _element_indexes(bl, sel)
  pos isa AbstractString && (pos = Symbol(pos))
  n_ele = length(bl.line)
  # (the end of the beamline is a zero-length element: one step from its beginning to its end,
  # which are the same location)
  nsteps(i) = i == n_ele + 1 ? 1 : _nsteps(bl, i)
  all_steps = pos isa Colon || pos == :all
  # (the locations of every placement)
  if all_steps
    per = [[_location(bl, i, k) for k in 0:(i == n_ele + 1 ? 0 : nsteps(i))] for i in idxs]
  elseif pos isa Integer
    per = [[_location(bl, i, pos)] for i in idxs]
  elseif pos == :exit
    per = [[_location(bl, i, nsteps(i))] for i in idxs]
  elseif pos == :entrance
    per = [[_location(bl, i, 0)] for i in idxs]
  else
    error("Unknown position $(repr(pos)): use :entrance, :exit, : or a step number")
  end
  if sel isa Integer && !all_steps
    return _single_location(bl, part, only(only(per)), strict), true
  elseif sel isa LatElement && strict
    return _one_placement(bl, part, sel, idxs, per), !all_steps
  end
  return filter!(l -> _in_part(bl, l, part), reduce(vcat, per; init=Tuple{Int,Int}[])), false
end

# The locations in the part of the only placement of the element sel that has locations there
# (idxs: the beamline indexes of its placements, per: their locations), else an error
function _one_placement(bl, part, sel, idxs, per)
  inpart = [filter(l -> _in_part(bl, l, part), p) for p in per]
  k = findall(!isempty, inpart)
  length(k) == 1 && return inpart[only(k)]
  n_ele = length(bl.line)
  where = part == (1, n_ele + 1) ? "the beamline" :
          "the part of the beamline from element $(part[1]) to element $(part[2]) (`start`, `stop`)"
  isempty(k) && error("Element $(repr(sel.name)) has no placement in ", where)
  is = idxs[k]
  error("Element $(repr(sel.name)) is placed $(length(k)) times in ", where, " (beamline indexes ",
        join(first(is, 10), ", "), length(is) > 10 ? ", ..." : "", "): index with `[[ele]]` for ",
        "all of them, or with one beamline index")
end

function _single_location(bl, part, l, strict)
  _in_part(bl, l, part) && return [l]
  strict && error("The location ", l[1] == -1 ? "at the end of the beamline" :
                  "$(l[2]) steps after the beginning of element $(l[1])", " is not in the part ",
                  "of the beamline from element $(part[1]) to element $(part[2]) (`start`, `stop`)")
  return Tuple{Int,Int}[]
end

# Every integration step of the part of the beamline whose s is in the range (range[1] <= s <
# range[2]), and the end of the part if range[1] <= s <= range[2] (all with range = nothing)
_range_locations(bl, part, range, ls) = _range_locations(bl, part, range, ls...)
function _range_locations(bl, (i1, i2), range, s, L, n, _...)
  isnothing(range) || range[1] <= range[2] || error("Invalid s range $range")
  inrange(x, upper) = isnothing(range) || (range[1] <= scalar(x) && upper(scalar(x), range[2]))
  locs = Tuple{Int,Int}[]
  for i in i1:i2-1, k in 0:n[i]-1
    inrange(_step_s(s, L, n, i, k), <) && push!(locs, (i, k))
  end
  inrange(s[i2], <=) && push!(locs, i2 == length(bl.line) + 1 ? (-1, 0) : (i2, 0))
  return locs
end

# The locations of `at` (of twiss and track) in the part of the beamline, ordered along it: each
# entry is one index of a column (see _column_locations), but beamline indexes outside the part are
# left out (ls: _line_steps(bl) if already computed)
function _at_locations(bl, part, at, ls=nothing)
  if at isa Colon
    locs = _range_locations(bl, part, nothing, isnothing(ls) ? _line_steps(bl) : ls)
  elseif at isa _Locations
    locs = filter(l -> _in_part(bl, l, part), at.v)
  else
    locs = Tuple{Int,Int}[]
    for x in at
      append!(locs, first(_column_locations(bl, part, x; strict=false, ls)))
    end
  end
  return sort!(unique!(locs); by=_location_key)
end

# The s, beamline index, name, kind, step from the beginning of the beamline and step in the
# element of the locations locs
_location_info(bl, locs, ls=_line_steps(bl)) = _location_info(locs, ls...)
function _location_info(locs, s, L, n, step0, ele_names, ele_kinds)
  n_ele = length(n)
  m = length(locs)
  ss = Vector{eltype(s)}(undef, m)
  names = Vector{String}(undef, m)
  kinds = Vector{String}(undef, m)
  steps = Vector{Int}(undef, m)
  for (j, (i, k)) in enumerate(locs)
    if i == -1
      ss[j] = s[n_ele+1]
      names[j] = "END"
      kinds[j] = "---"
      steps[j] = step0[n_ele+1]
    else
      ss[j] = _step_s(s, L, n, i, k)
      names[j] = ele_names[i]
      kinds[j] = ele_kinds[i]
      steps[j] = step0[i] + k
    end
  end
  return ss, Int[l[1] for l in locs], names, kinds, steps, Int[l[2] for l in locs]
end
