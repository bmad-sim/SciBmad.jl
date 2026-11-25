# overview(bl): a compact summary of a SciBmad Beamline, for the start of any task.
# Usage: include(joinpath(<skill dir>, "scripts", "overview.jl")); overview(ring)
# Prints: reference particle, length, elements by kind, families (shared parents and context
# variables of DefExprs), misalignments, RF/radiation, tunes and chromaticities. Read-only.
using SciBmad

const _STRENGTH = Dict("Quadrupole" => :Kn1, "Sextupole" => :Kn2, "Octupole" => :Kn3,
                       "SBend" => :Kn0, "RFCavity" => :voltage)

_short(v) = v isa AbstractVector ? "vector[$(length(v))]" : repr(v isa AbstractFloat ? round(v, sigdigits=6) : v)

function overview(bl::Beamline; optics::Bool=true)
  line = bl.line
  println("Reference: ", nameof(bl.species_ref), ", pc_ref = ", bl.pc_ref, " eV, E_ref = ", bl.E_ref,
          " eV; length ", round(bl[end].s_downstream, sigdigits=8), " m, ", length(line), " elements")
  kinds = Dict{String,Int}(); for e in line; kinds[e.kind] = get(kinds, e.kind, 0) + 1; end
  println("Kinds: ", join(["$k $n" for (k, n) in sort(collect(kinds), by=x -> -x[2])], ", "))
  for (kind, p) in _STRENGTH
    es = [e for e in line if e.kind == kind]; isempty(es) && continue
    parents = unique(e -> objectid(e.parent), es)
    nde = count(e -> (try rawget(e, p) isa DefExpr catch; false end), es)
    fams = [string(f.name, "×", count(e -> e.parent === f.parent, es)) for f in parents]
    println(kind, ": ", length(es), " placements of ", length(parents), " elements",
            nde > 0 ? ", $nde with DefExpr $p" : "", length(fams) <= 12 ? " (" * join(fams, ", ") * ")" : "")
  end
  ctx = bl.context; vars = collect(propertynames(ctx))
  if !isempty(vars)
    println("Context variables: ", join(["$v = $(_short(getproperty(ctx, v)))" for v in vars], ", "))
    drive = Dict{String,Vector{String}}()
    for v in vars
      x = getproperty(ctx, v); x isa Union{Number,AbstractVector{<:Number}} || continue
      for k in (x isa AbstractVector ? eachindex(x) : (0,))
        c2 = copy(ctx)
        if k == 0; setproperty!(c2, v, x + 1e-3) else y = copy(x); y[k] += 1e-3; setproperty!(c2, v, y) end
        moved = String[]
        for e in line
          p = get(_STRENGTH, e.kind, nothing); p === nothing && continue
          r = try rawget(e, p) catch; nothing end
          r isa DefExpr && deval(r, c2) != getproperty(e, p) && push!(moved, e.name)
        end
        isempty(moved) || (drive[k == 0 ? "$v" : "$v[$k]"] = unique(moved))
      end
    end
    for (v, m) in sort(collect(drive)); println("  ", v, " drives ", join(first(m, 8), ", "), length(m) > 8 ? ", … ($(length(m)))" : ""); end
  end
  mis = count(e -> any(p -> (try !iszero(getproperty(e, p)) catch; false end), (:x_offset, :y_offset, :tilt, :x_rot, :y_rot)), line)
  rad = count(e -> e.rad_damping, line); fl = count(e -> e.rad_fluctuations, line)
  rf = count(e -> e.kind in ("RFCavity", "CrabCavity") && (try !iszero(e.voltage) catch; false end), line)
  println("Misaligned elements: $mis; rad_damping: $rad, rad_fluctuations: $fl; RF cavities with voltage: $rf")
  optics || return nothing
  try
    tw = twiss(bl, at=[])
    println("Tunes: q1 = ", tw.q1, ", q2 = ", tw.q2, :q3 in propertynames(tw) ? ", q3 = $(tw.q3)" : "")
    tc = twiss(bl, at=[], chrom=2, rf_on=false)
    println("Chromaticities (rf_on=false): dq1 = ", tc.dq1, ", dq2 = ", tc.dq2, "; alphac = ", tc.alphac[delta=0])
  catch err
    println("Periodic optics failed (open line or unstable?): ", first(split(sprint(showerror, err), '\n')))
  end
  return nothing
end
