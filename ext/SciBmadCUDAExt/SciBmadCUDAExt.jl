module SciBmadCUDAExt
using CUDA
using CUDSS # Sparse CUDA matrix solving on the device
import SciBmad

SciBmad.device_specific_sparse(m, device::CUDA.CUDABackend) = CUSPARSE.CuSparseMatrixCSR(m)

end