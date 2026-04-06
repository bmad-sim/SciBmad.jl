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

  else
    error("Invalid batchdim (must be either 1, 2, or nothing)")
  end
end

end