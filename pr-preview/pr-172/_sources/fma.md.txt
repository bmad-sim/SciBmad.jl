---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Julia
  language: julia
  name: julia
---

(fma)=
# Frequency Map Analysis

```{code-cell} julia
:tags: [remove-cell]
using SciBmad
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

Frequency map analysis finds the tunes of tracked particles from their coordinates turn by turn.
The function `naff` (Numerical Analysis of Fundamental Frequencies) computes the main frequencies
of signals, one signal per row of a matrix. For example, the horizontal tunes of four particles
at different amplitudes in a FODO cell with sextupoles:

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=20.0, L=0.2)
  d = Drift(L=1.0)
  b = SBend(L=2.0, angle=pi/20)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-30.0, L=0.2)
end
fodo = Beamline([qf, sf, d, b, d, qd, sd, d, b, d], species_ref=Species("electron"),
                pc_ref=18e9)

v0 = zeros(4, 6)
v0[:, 1] = [1e-4, 1e-3, 3e-3, 6e-3] # x in m
res = track(fodo, v0=v0, n_turns=1023)

tw = twiss(fodo)
β, α = tw.beta1[1], tw.alpha1[1]
x, px = res.x[turn=:], res.px[turn=:] # [particle, turn]
q, amp = naff(x .- im .* (β .* px .+ α .* x))
q # 4×1 matrix: 0.15497, 0.15497, 0.15493, 0.15483
```

Each row of the matrix is the signal of one particle, from turn 0 to turn 1023. The complex
signal $x - i(\beta p_x + \alpha x)$, with the Twiss parameters at the start, turns in one
direction in the complex plane, so its frequency is the tune including its sign. A real signal,
such as `x` alone, has the frequencies $\pm q$, and either one can be found. The signals must
not have a constant part: subtract the closed orbit if it is not zero (`naff` warns otherwise).

At small amplitude, the tune agrees with the tune `q1` of `twiss`, here to 5e-8:

```{code-cell} julia
q[1], tw.q1 # (0.1549708764, 0.1549709220)
```

At larger amplitudes the tune decreases. This is the detuning with amplitude, which `twiss` gives
as the derivatives of `q1` with respect to the action `J₁` (see
[Chromatic and amplitude-dependent quantities](#twiss.orders)).

`naff(data, n)` computes the first `n` frequencies of each signal, and their complex amplitudes.
The algorithm works on all signals at once, so with the data on a GPU (e.g. a `CUDA.CuArray`),
all signals are analyzed in parallel there.

```{docstring} naff
```
