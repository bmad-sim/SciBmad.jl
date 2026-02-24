module SciBmadCUDAExt
using CUDA
using CUDSS # Sparse CUDA matrix solving on the device
import SciBmad: default_solver

function default_solver(::CUDA.CUDABackend, _batchsize)
  let batchsize=_batchsize 
    return (dx, jac, y)-> begin
      hi
      

    end
  end
end

end