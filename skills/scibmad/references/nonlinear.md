# Nonlinear dynamics: driving terms, detuning, dynamic aperture optimization

For tasks like "optimize the dynamic aperture" or "which resonance driving terms / detuning
coefficients matter most". Builds on `twiss.md` (h columns, orders) and `optimize.md`
(`match!`, `minimize`). Keep it cheap: these calculations are the expensive ones.

## Contents
1. Cost rules
2. Computing the terms
3. Ranking what matters
4. Optimizing the dynamic aperture
5. Diagnosing a result

## 1. Cost rules

- Everything nonlinear needs a coasting beam: `rf_on=false` (in `twiss` and `TwissExpr`).
- An `h` column at a row is the one-turn term seen from that row. For inspecting or ranking,
  one row (`at=[1]`) is enough. For optimizing the dynamic aperture, prefer the s-dependent
  terms (section 4): rows at every sextupole, not more.
- Use the lowest order that resolves a term: `h<ijkl>` needs `order ≥ i+j+k+l−1` (3rd-order
  terms such as `h3000`, `h2100`, `h1020`, `h1002`, `h1011`: `order=2`; 4th-order terms such as
  `h2200`, `h1111`, `h0022`, `h4000`: `order=3`); `q1[J1=1]` needs `order=3`, `q1[J1=2]`
  `order=5`; `q1[delta=n]` needs `chrom=n+1`. Cost grows quickly with the order.
- `TwissExpr` picks orders and rows itself and computes only what the expressions need.
- `dynamic_aperture` is the most expensive step: start with few turns and a coarse grid
  (`n_turns=200`, `n_r≈11`, `n_theta≈7`, 2–3 `deltas`), then refine for the final answer. Use
  `coordinates_number_type=Float32` and a GPU `backend` for large runs.

## 2. Computing the terms

```julia
cols = ["h3000", "h2100", "h1020", "h1002", "h1011",          # 3rd order (sextupole-like)
        "h4000", "h2200", "h1111", "h0022", "h3100", "h2020"]  # 4th order
tw = twiss(ring, order=3, rf_on=false, at=[1], cols=cols)
tw.h2100[1]                                    # complex
tw.q1[J1=1], tw.q1[J2=1], tw.q2[J2=1]          # amplitude detuning ∂q/∂J (normal-form actions)
twc = twiss(ring, chrom=3, rf_on=false, cols=["w1", "w2"])
twc.dq1, twc.dq1_2                             # chromaticity and its δ-derivative; W functions
```

Chromatic derivatives of driving terms are columns too (`dh2000`, …, coasting beam).
Detuning is best read directly from the tunes (`q1[J1=1]`, …) rather than converted from
`h2200`-type terms.

## 3. Ranking what matters

A term's absolute size is not its importance. Weigh each by:

- **Amplitude where it acts:** a term of order n (= i+j+k+l) contributes like
  |h| (2J₁)^{(i+j)/2} (2J₂)^{(k+l)/2}. Evaluate at the target amplitude, with
  J_k = x²/(2β_k) at the row (e.g. x = the DA goal, n_σ · σ_x).
- **Distance from its resonance:** `h_ijkl` with i ≠ j or k ≠ l drives the resonance
  (i−j)Q₁ + (k−l)Q₂ = integer; to first order its effect on the motion goes like
  |h| / |2 sin(π((i−j)Q₁ + (k−l)Q₂))|, with the tunes including their dependence on the
  amplitude (detuning can move the tunes onto a resonance at large amplitude).
- **Detuning:** compute the tune shift at the target amplitude,
  ΔQ₁ ≈ q1[J1=1]·J₁ + q1[J2=1]·J₂, and check which resonance lines the tune footprint crosses.
- **Chromatic terms** (`dq1_2`, W functions, `dh…`) matter for the off-momentum DA (the
  `deltas` of `dynamic_aperture`).

Confirm a ranking by experiment, not only by formulas: reduce one term (a `match!` with only
that term in `minimize`, chromaticity held) and see whether the DA or the FMA changes.
Sensitivities of a term to each sextupole: `SciBmad.value_and_jacobian([real(te.h2100[1]),
imag(te.h2100[1])], knobs)` or `deriv`.

## 4. Optimizing the dynamic aperture

The DA is not differentiable; optimize smooth proxies with `match!` (exact derivatives), and
use the DA only to judge the result.

1. **Baseline.** Tunes, chromaticity (`rf_on=false`), DA with cheap settings. If the DA is
   already beyond the physical aperture (or the requested goal), say so and stop. If results look
   odd, first check convergence in `n_steps` of the strong/long elements.
2. **Find the existing sextupole families first** and use them as knobs rather than inventing
   new ones. Read the lattice file: the families are usually context variables behind
   `DefExpr`s (e.g. `Kn2 = DefExpr(c -> c.k2f + c.xf[i])`), and the file shows them with their
   names, trims and intent (`grep -n "DefExpr\|context\.\|Sextupole(" ring.jl`). Vary those
   variables (`@knob(ring.context.k2f)`). Only if there is no source (the lattice was built
   elsewhere or passed in as an object), find which sextupoles each variable drives without
   touching the lattice:

   ```julia
   ctx = ring.context; sxs = ring["Sextupole::*"]
   for v in propertynames(ctx)
     x = getproperty(ctx, v)
     for k in (x isa AbstractVector ? eachindex(x) : (0,))
       c2 = copy(ctx)       # changed copy; the lattice is not modified
       if k == 0; setproperty!(c2, v, x + 1e-3) else; y = copy(x); y[k] += 1e-3; setproperty!(c2, v, y) end
       moved = [e.name for e in sxs if rawget(e, :Kn2) isa DefExpr && deval(rawget(e, :Kn2), c2) != e.Kn2]
       isempty(moved) || println(v, k == 0 ? "" : "[$k]", " -> ", moved)
     end
   end
   ```

   (Variables kept only in `DEFAULT_CONTEXT` are not in `propertynames(ring.context)`.)
   Sextupoles with plain numbers form families through shared parents: placements of one
   element (`unique(e -> e.parent, sxs)`) move together, `@knob(sf.Kn2)`. Otherwise use the
   names (`ring["sf*"]`). Ask before splitting families: that changes the machine.
3. **Knobs that keep the chromaticity.** Sextupole families as context variables, split so
   that some combinations leave the chromaticity unchanged, e.g. per arc
   `Kn2 = DefExpr(c -> c.k2f + c.xf[i])` and `DefExpr(c -> c.k2f - c.xf[i])` for two sextupoles of
   one family (the ring design tutorial, chapter 12). Phase advances between arcs can be knobs
   too: phase trombones with context phases (`lattice.md`), placed so that their sum is zero
   (`-c.x1 - c.x2` on a third one) to keep the tunes.
4. **One match with weighted proxies, using s-dependent terms.** Minimize the driving terms at
   every sextupole (a vector expression per term), not only the one-turn value at one point:
   this controls the local cancellation between sextupoles, which is more robust for the DA.

```julia
te = TwissExpr(ring; rf_on=false)
J1, J2 = 30e-9, 30e-9                        # actions at the target amplitude [m]
at = "Sextupole::*"                          # rows: every sextupole placement
h3 = [te.h2100[at] * (2J1)^1.5, te.h3000[at] * (2J1)^1.5,
      te.h1020[at] * sqrt(2J1)*2J2, te.h1002[at] * sqrt(2J1)*2J2]
mn = LatticeExpr[]
for h in h3
  append!(mn, real(h)); append!(mn, imag(h))   # real and imaginary parts as separate entries
end
append!(mn, [te.q1[J1=1]*J1, te.q1[J2=1]*J2, te.q2[J2=1]*J2,
             te.dq1_2 * 1e-4, te.dq2_2 * 1e-4])  # chromatic terms scaled by δ² of interest
match!(vary = knobs,
       targets = [Target(te.dq1, 1; weight=100, tol=1e-4), Target(te.dq2, 1; weight=100, tol=1e-4)],
       minimize = mn)
```

   Minimize (or target) the real and imaginary parts of each complex term as separate entries,
   never `abs(h)`: then h = 0 is a smooth root of both parts, which Gauss-Newton converges to,
   instead of the kink of |h| at 0. `minimize` takes a flat list of expressions: a vector
   expression nested in the list fails with a misleading "targets cannot be computed" error. The amplitude factors make the terms comparable; the chromaticity targets get a
   large weight so they hold. Detuning and chromatic terms are one-turn quantities (no row).
   Add `w1a`/`w1b`/`w2a`/`w2b` at an IP to correct the W functions.
   **Keep the sextupoles moderate:** many trims can cancel the terms with strengths several
   times the family values, which drives higher-order terms and shrinks the DA. Bound the trims
   (`@knob(ctx.tf, lower=-a, upper=a)` with `a` a fraction of the family strength), or add the
   trim values to `minimize` with a small weight, and judge every result by the DA.
5. **Evaluate** the DA (on- and off-momentum). Adjust which terms and weights are minimized
   based on what limits it (section 5), and repeat. Each `match!` is one `twiss` per iteration
   plus derivatives, far cheaper than DA evaluations.
6. **Only if needed**, a derivative-free global search directly on the DA (e.g.
   Metaheuristics on a function that calls `dynamic_aperture`), on a GPU, with few knobs: it is
   very expensive. Start it from the proxy optimum, and polish proxies again with `match!`.

## 5. Diagnosing a result

- **Where particles are lost:** `dynamic_aperture(...; output_file=...)` writes the grid with
  each particle's final state; `track` with `at=[...]` shows where (`res.state`).
- **Which resonance:** a frequency map. Track a grid of amplitudes for 2N turns, `naff` the
  first and second N turns separately (complex signal `x - i(βpx + αx)`, `tracking.md`), and
  plot q₁ vs q₂ coloured by log₁₀|Δq| (tune diffusion). Particles locked on a line
  m₁Q₁ + m₂Q₂ = p point to the driving term with (i−j, k−l) = (m₁, m₂).
- **Is it the model?** Increase `n_steps` in sextupoles and quadrupoles, and the integrator
  order, before trusting a DA difference of a few percent.
