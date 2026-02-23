# Temporary type piracy solution to https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/970
module SciBmadGPUArraysCoreExt
import DifferentiationInterface as DI
using GPUArraysCore: @allowscalar, AbstractGPUArray
using LinearAlgebra: Transpose

function DI.basis(a::Transpose{<:AbstractGPUArray{T}}, i) where {T}
    b = similar(a)
    fill!(b, zero(T))
    @allowscalar b[i] = oneunit(T)
    return b
end

function DI.multibasis(a::Transpose{<:AbstractGPUArray{T}}, inds) where {T}
    b = similar(a)
    fill!(b, zero(T))
    view(b, inds) .= oneunit(T)
    return b
end

end