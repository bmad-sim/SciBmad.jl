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

    let pivot=_pivot, info=_info, batchsize=_batchsize, n=_n
      return (dx, jac, y)-> begin
        jacs = reshape(jac.nzVal, n, n, batchsize)
        ys = reshape(y, n, 1, batchsize)
        CUBLAS.getrf_strided_batched(jacs, pivot, info)
        CUBLAS.getrs_strided_batched('N', jacs, ys, pivot)
        dx .= reshape(ifelse.(reshape(info, 1, batchsize) .!= 0, NaN32, reshape(-y, n, batchsize)), :)
        #dx .= -y
      end
    end
  elseif batchdim == 1
    #= Claude proposes:
       _batchsize = size(_x, 1)
    _n = size(_y, 2)
    _pivot = CUDA.zeros(Int32, _n, _batchsize)
    _info = CUDA.zeros(Int32, _batchsize)
    _jac_dense = CUDA.zeros(eltype(_y), _n, _n, _batchsize)
    _rhs = CUDA.zeros(eltype(_y), _n, 1, _batchsize)
    let pivot=_pivot, info=_info, batchsize=_batchsize, n=_n, jac_dense=_jac_dense, rhs=_rhs
      return (dx, jac, y) -> begin
        nzval_3d = reshape(jac.nzval, n, batchsize, n)  # (n_rows, batchsize, n_cols)
        permutedims!(@view(jac_dense[:, :, :]), nzval_3d, (1, 3, 2))  # → (n_rows, n_cols, batchsize)
        rhs .= reshape(y, n, 1, batchsize)
        CUBLAS.getrf_strided_batched(jac_dense, pivot, info)
        CUBLAS.getrs_strided_batched('N', jac_dense, rhs, pivot)
        dx .= reshape(ifelse.(reshape(info, 1, batchsize) .!= 0, NaN32, -reshape(rhs, n, batchsize)), :)
      end
    end
  =#

  else
    error("Invalid batchdim (must be either 1, 2, or nothing)")
  end
end

end