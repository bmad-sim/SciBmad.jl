# Parametric Normal Form

All analysis (e.g. computing amplitude-dependent tunes, nonlinear Twiss parameters, invariant spin field, etc.) is done using SciBmad's [`NonlinearNormalForm` package](https://github.com/bmad-sim/NonlinearNormalForm.jl). 

SciBmad also provides parametric normal form calculations, for example computing how the tunes depend on magnet strengths or how the invariant spin field depends on a misalignment. To do so, knowledge of the underlying truncated power series algebra package [`GTPSA`](https://github.com/bmad-sim/GTPSA.jl) is highly useful. We recommend first going through the [GTPSA quickstart guide](https://bmad-sim.github.io/GTPSA.jl/stable/quickstart/) first to gain an understanding of the high order autodiff techniques, before going through this section.

Let's compute how the tunes depend on the quadrupole strengths in our FODO cell example.

```@example nf
using SciBmad # hide

@elements begin
  qf = Quadrupole(Kn1=DefExpr(c -> c.kq), L=0.5)
  sf = Sextupole(Kn2=DefExpr(c-> c.sf), L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=DefExpr(c -> c.kd), L=0.5)
  sd = Sextupole(Kn2=DefExpr(c -> c.sd), L=0.2)
end

fodo = Beamline([qf, sf, b, d, qd, sd, b, d], 
        species_ref=Species("electron"), pc_ref=18e9)

fodo.context.kq = 0.36
fodo.context.kd = -0.36
fodo.context.sf = 1.2
fodo.context.sd = -1.2
```

To do so, we will define a GTPSA `Descriptor` with two parameters, corresponding to infinitesimal variations in each of the quadrupoles. Because computing the tunes requires only 1st order in the phase space variables, we can save computation time by explicitly specifying the truncation orders for the variables to be 1, truncate the parameters part also at 1, but allow maximum order of 2:

```@example nf
dnf = Descriptor([1, 1, 1, 1, 1, 1], 2, [1, 1], 1)
dk = params(dnf)
fodo.context.kq += dk[1]
fodo.context.kd += dk[2]
```

Now we can just use `twiss` through the usual machinery, though this time provide the descriptor explicitly to the `GTPSA_descriptor` keyword argument

```@example nf
tw = twiss(fodo, GTPSA_descriptor=dnf)
```

By explicitly providing a GTPSA descriptor including parameter dependence, the `as_taylor_series` keyword argument to `twiss` is automatically set to true. Therefore, the columns are now Taylor series.

We also see the amplitude-dependent tunes are printed as Taylor series.