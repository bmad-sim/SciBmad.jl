module SciBmadCUDAExt
using CUDA
import SciBmad: default_solver

function default_solver(device::CUDA.CUDABackend, _y, _x, ::Val{_batched}) where {_batched}
  _lx = length(_x)
  _ly = length(_y)

  # Preallocate stuff to include in the returned closure
  # If not batch, then just do regular dense matrix solver
  if !_batched
    let lx=_lx, ly=_ly
      return (dx, jac, y)->(reshape(dx, lx) .= -jac \ reshape(y, ly))
    end
  end

  # Batch:
  if _ly != _lx
    error("CUDA batched matrix solver for non-square systems will be implemented soon.")
  end

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
      CUBLAS.getrf_strided_batched!(jacs, pivot, info)
      CUBLAS.getrs_strided_batched!('N', jacs, ys, pivot)
      dx .= -y
    end
  end
end

end