grad(t::TPS) = GTPSA.gradient(t, include_params=true)[7:end]
grad(t::AmplitudeDependentValue) = grad(t.taylor_series)
jac(t::AbstractArray{<:TPS}) = GTPSA.jacobian(t, include_params=true)[:,7:end]
val(t::Number) = scalar(t)
val(t::AbstractArray{<:Number}) = scalar.(t)