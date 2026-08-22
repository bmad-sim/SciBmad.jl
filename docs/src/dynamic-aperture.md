(dynamicaperture)=
# Dynamic Aperture

The `dynamic_aperture` function computes the acceptance of a ring by pushing a polar grid in
$x/\sigma_x$ and $y/\sigma_y$ space, for each of the provided `deltas`. It returns a tuple of
two vectors defining the first particle loss along the radius for a given angle on the polar
grid, where the first index corresponds to the line position in $x/\sigma_x$ or
$y/\sigma_y$ space, and the second index corresponds to that in `deltas`.

The required keyword arguments are the polar grid resolution (`n_r`, `n_theta`), its extent
(`max_sig_x`, `max_sig_y`), the assumed emittances (`emit_1`, `emit_2`), the `deltas` to
scan, and `n_turns`. A `backend` keyword argument selects CPU or GPU execution — pass
`CUDA.CUDABackend()` to run the whole scan on a CUDA GPU.

Complete, worked examples with plots are available as runnable notebooks:

- [Dynamic aperture](examples/julia/dynamic-aperture.ipynb) — a full CPU scan of a ring,
  including plotting the resulting aperture.
- [Dynamic aperture (GPU)](examples/julia/dynamic-aperture-gpu.ipynb) — the same scan
  GPU-parallelized.

:::{seealso}
The full `dynamic_aperture` docstring, listing every keyword argument, is in the
{external:doc}`API Reference <index>`.
:::
