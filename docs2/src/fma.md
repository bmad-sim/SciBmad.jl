# Frequency Map Analysis

Frequency map analysis can be performed using the exported `naff` function, which accepts an input array of `n_particles x n_samples`. The entire NAFF algorithm implemented in SciBmad is fully GPU-vectorizable, and so NAFF can be performed in parallel on the GPU if the input array is a `CUDA.CuArray`.

```@docs
naff
```