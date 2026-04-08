module SciBmadCUDAExt
using CUDA
import SciBmad: default_solver

function default_solver(device::CUDA.CUDABackend, _y, _x, batchdim::Integer)
  _lx = length(_x)
  _ly = length(_y)

  # Batch:
  if _ly != _lx
    error("CUDA batched matrix solver for non-square systems will be implemented soon.")
  end

  if batchdim == 2
    # Each element of a batch is a COLUMN
    # Number of rows = number of variables in an element of a batch
    _batchsize = size(_x, 2)
    _n = size(_x, 1)
    _pivot = CUDA.zeros(Int32, _n, _batchsize)
    _info = CUDA.zeros(Int32, _batchsize)
    _jacscratch = CUDA.zeros(eltype(_y), _n, _n, _batchsize)

    let pivot=_pivot, info=_info, batchsize=_batchsize, n=_n, jacscratch=_jacscratch
      return (dx, jac, y)-> begin
        jacsratch .= reshape(jac.nzVal, n, n, batchsize)
        ys = reshape(y, n, 1, batchsize)
        CUBLAS.getrf_strided_batched!(jacscratch, pivot, info)
        CUBLAS.getrs_strided_batched!('N', jacscratch, ys, pivot)
        dx .= reshape(ifelse.(reshape(info, 1, batchsize) .!= 0, NaN32, -reshape(y, n, batchsize)), :)
      end
    end
  elseif batchdim == 1
    _batchsize = size(_x, 1)
    _n = size(_y, 2)
    _pivot = CUDA.zeros(Int32, _n, _batchsize)
    _info = CUDA.zeros(Int32, _batchsize)
    _jacscratch = CUDA.zeros(eltype(_y), _n, _n, _batchsize)
    _rhs = CUDA.zeros(eltype(_y), _n, 1, _batchsize)

    let pivot=_pivot, info=_info, batchsize=_batchsize, n=_n, jacscratch=_jacscratch, rhs=_rhs
      return (dx, jac, y) -> begin
        nzval_3d = reshape(jac.nzVal, n, batchsize, n)  # (n_rows, batchsize, n_cols)
        permutedims!(jacscratch, nzval_3d, (1, 3, 2))  # → (n_rows, n_cols, batchsize)
        @show jacscratch
        # Also need to permute y dims from (batchsize, 1, n_rows) to (n_rows, 1, batchsize)
        permutedims!(rhs, reshape(y, batchsize, 1, n), (3, 2, 1))
        @show pivot
        @show rhs
        CUBLAS.getrf_strided_batched!(jacscratch, pivot, info)
        @show pivot
        CUBLAS.getrs_strided_batched!('N', jacscratch, rhs, pivot)
        @show rhs
        @show pivot
        # Now need to permutedims back
        permutedims!(reshape(y, batchsize, 1, n), rhs, (3, 2, 1))
        @show y
        # ready to go
        dx .= ifelse.(reshape(info, batchsize, 1) .!= 0, NaN32, -reshape(y, batchsize, n))
      end
    end
  else
    error("Invalid batchdim (must be either 1, 2, or nothing)")
  end
end

end