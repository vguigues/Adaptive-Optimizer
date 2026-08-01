using LinearAlgebra

aux_norm = (x::AbstractArray) -> return sqrt(sum(abs2, x))

function adaptive_optimizer(x0::AbstractArray{Float64}, simulator::Function, iters::AbstractVector, D::Float64, atest::AbstractMatrix, ytest::AbstractArray, compute_accuracy::Function)
	n=size(x0)
	x=copy(x0)
	s=zeros(size(x0));
	ssq=0
	fs=zeros(iters[end])
	xsum=zeros(n)
	acs=[]

	for i in 1:iters[end]
		f, g=simulator(x)
		fs[i]=f
		s+=g
		ssq+=sqrt(sum(abs2, g))
		gamma=D*sqrt(sum(abs2, s))/ssq
		x=x0-gamma*s
		ac1=compute_accuracy(x, ytest, atest)
		push!(acs, ac1)
		xsum+=x
	end
	x.=xsum/iters[end]
	return x, fs, acs
end

function adaptive_optimizer_cwise(x0::AbstractArray{Float64}, simulator::Function, iters::AbstractVector, D::Float64, epsilon::Float64, atest::AbstractMatrix, ytest::AbstractArray, compute_accuracy::Function)
	n=size(x0, 1)
	x=copy(x0)
	s=zeros(size(x0))
	ssq=zeros(size(x0))
	gamma=zeros(size(x0))
	fs=zeros(iters[end])
	xsum=zeros(size(x0))
	acs=[]
	for i in 1:iters[end]
		f, g=simulator(x)
		fs[i]=f
		s+=g
		ssq.+=g .^ 2
		gamma=D*abs.(s) ./ (ssq .+ epsilon)
		x=x0-gamma .* s
		xsum+=x
		ac1=compute_accuracy(x, ytest, atest)
		# println("Adaptive optimizer iteration $i, accuracy: $ac1")
		push!(acs, ac1)
	end
	x.=xsum/iters[end]

	return x, fs, acs
end

function adagrad(x0::AbstractArray{Float64}, simulator::Function, iters::AbstractVector, eta::Float64, epsada::Float64, atest::AbstractMatrix, ytest::AbstractArray, compute_accuracy::Function)
	n=size(x0)
	x=copy(x0)
	ssq=zeros(size(x0))
	fs=zeros(iters[end])
	acs=[]

	for i in 1:iters[end]
		f, g=simulator(x)
		fs[i]=f
		ssq.+=g .^ 2
		x=x-eta*g ./ sqrt.(epsada .+ ssq)
		ac1=compute_accuracy(x, ytest, atest)
		# println("Adagrad iteration $i, accuracy: $ac1")
		push!(acs, ac1)
	end
	return x, fs, acs
end

function ema(x0::AbstractArray{Float64}, simulator::Function, iters::AbstractVector, beta::Float64, gamma::Float64, epsilon::Float64, beta3::Float64, atest::AbstractMatrix, ytest::AbstractArray, compute_accuracy::Function)
	x=copy(x0)
	n=size(x0)
	s=zeros(size(x0))
	m=zeros(size(x0))
	fs=zeros(iters[end])
	v=0
	acs=[]
	for i in 1:iters[end]
		f, g=simulator(x)
		fs[i]=f
		s=beta*s+(1-beta)*g
		v=beta*v+(1-beta)*aux_norm(g)^2
		m=beta3*m+(1-beta3)*g
		x-=gamma*(aux_norm(s)/(v+epsilon))*m
		ac1=compute_accuracy(x, ytest, atest)
		push!(acs, ac1)
	end
	return x, fs, acs
end

function emawise(x0::AbstractArray{Float64}, simulator::Function, iters::AbstractVector, beta::Float64, gamma::Float64, epsilon::Float64, beta3::Float64, atest::AbstractMatrix, ytest::AbstractArray, compute_accuracy::Function)
	x=copy(x0)
	n=size(x0)
	s=zeros(n)
	fs=zeros(iters[end])
	v=zeros(size(x0))
	m=zeros(size(x0))
	acs=[]
	for i in 1:iters[end]
		f, g=simulator(x)
		fs[i]=f
		s=beta*s+(1-beta)*g
		m=beta3*m+(1-beta3)*g
		v=beta*v+(1-beta)*(g .^ 2)
		x-=gamma*((abs.(s) ./ (v .+ epsilon)) .* m)
		ac1=compute_accuracy(x, ytest, atest)
		push!(acs, ac1)
	end
	return x, fs, acs
end

function adam(x0::AbstractArray{Float64}, simulator::Function, iters::AbstractVector, beta1::Float64, beta2::Float64, epsilon::Float64, alpha::Float64, atest::AbstractMatrix, ytest::AbstractArray, compute_accuracy::Function)
	x=copy(x0)
	n=size(x0)
	m=zeros(size(x0))
	v=zeros(size(x0))
	fs=zeros(iters[end])
	acs=[]
	for i in 1:iters[end]
		f, g=simulator(x)
		fs[i]=f
		m=beta1*m+(1-beta1)*g
		v=beta2*v+(1-beta2)*(g .^ 2)
		mhat=m/(1-(beta1^i))
		vhat=v/(1-(beta2^i))
		x-=alpha*mhat ./ (sqrt.(vhat) .+ epsilon)
		ac1=compute_accuracy(x, ytest, atest)
		push!(acs, ac1)
	end
	return x, fs, acs
end

function _optimizer_iterations(iters::AbstractVector)
	isempty(iters) && throw(ArgumentError("iters must not be empty"))
	niters = iters[end]
	niters isa Integer || throw(ArgumentError("iters[end] must be an integer"))
	niters > 0 || throw(ArgumentError("iters[end] must be positive"))
	return niters
end

function _check_probability(value::Float64, name::String)
	0.0 <= value < 1.0 || throw(ArgumentError("$name must satisfy 0 <= $name < 1"))
end

"""
		prodigy_sgd(x0, simulator, iters, d0, G, atest, ytest, compute_accuracy)

Prodigy gradient-descent variant (Algorithm 1 of Mishchenko and Defazio).
`d0` is the initial positive lower bound on the distance to a solution and `G`
is an optional gradient-norm bound (use `0.0` when it is unknown).
"""
function prodigy_sgd(x0::AbstractArray{Float64}, simulator::Function,
		iters::AbstractVector, d0::Float64, G::Float64,
		atest::AbstractMatrix, ytest::AbstractArray,
		compute_accuracy::Function)
	d0 > 0.0 || throw(ArgumentError("d0 must be positive"))
	G >= 0.0 || throw(ArgumentError("G must be non-negative"))
	niters = _optimizer_iterations(iters)

	x = copy(x0)
	d = d0
	weighted_gradient_norm = 0.0
	d_numerator = 0.0
	average_numerator = zeros(size(x0))
	average_denominator = 0.0
	fs = zeros(niters)
	acs = Float64[]

	for i in 1:niters
		f, g = simulator(x)
		fs[i] = f
		weighted_gradient_norm += d^2 * sum(abs2, g)
		denominator = d^2 * G^2 + weighted_gradient_norm
		eta = d^2 / max(sqrt(denominator), eps(Float64))

		average_numerator .+= eta .* x
		average_denominator += eta
		d_numerator += eta * dot(vec(g), vec(x0 .- x))
		x .-= eta .* g

		d_hat = d_numerator / max(aux_norm(x .- x0), eps(Float64))
		d = max(d, d_hat)
		push!(acs, compute_accuracy(x, ytest, atest))
	end

	return average_numerator ./ average_denominator, fs, acs
end

"""
		prodigy_adam(x0, simulator, iters, d0, beta1, beta2, epsilon,
		             atest, ytest, compute_accuracy; gamma=1.0)

Adam variant of Prodigy (Algorithm 3). `gamma` is the external learning-rate
multiplier; the paper uses `1.0` (optionally with a scheduler).
"""
function prodigy_adam(x0::AbstractArray{Float64}, simulator::Function,
		iters::AbstractVector, d0::Float64, beta1::Float64, beta2::Float64,
		epsilon::Float64, atest::AbstractMatrix, ytest::AbstractArray,
		compute_accuracy::Function; gamma::Float64=1.0)
	d0 > 0.0 || throw(ArgumentError("d0 must be positive"))
	_check_probability(beta1, "beta1")
	_check_probability(beta2, "beta2")
	epsilon > 0.0 || throw(ArgumentError("epsilon must be positive"))
	gamma > 0.0 || throw(ArgumentError("gamma must be positive"))
	niters = _optimizer_iterations(iters)

	x = copy(x0)
	d = d0
	m = zeros(size(x0))
	v = zeros(size(x0))
	s = zeros(size(x0))
	r = 0.0
	sqrt_beta2 = sqrt(beta2)
	fs = zeros(niters)
	acs = Float64[]

	for i in 1:niters
		f, g = simulator(x)
		fs[i] = f
		m .= beta1 .* m .+ (1.0 - beta1) .* d .* g
		v .= beta2 .* v .+ (1.0 - beta2) .* d^2 .* (g .^ 2)
		r = sqrt_beta2 * r +
			(1.0 - sqrt_beta2) * gamma * d^2 * dot(vec(g), vec(x0 .- x))
		s .= sqrt_beta2 .* s .+
			(1.0 - sqrt_beta2) .* gamma .* d^2 .* g

		d_hat = r / max(sum(abs, s), eps(Float64))
		x .-= gamma .* d .* m ./ (sqrt.(v) .+ d * epsilon)
		d = max(d, d_hat)
		push!(acs, compute_accuracy(x, ytest, atest))
	end

	return x, fs, acs
end

"""
		bcos(x0, simulator, iters, eta, beta, epsilon, mode,
		     atest, ytest, compute_accuracy)

BCOS without weight decay. `mode` may be `:g` (gradient/RMSprop), `:m`
(momentum with EMA second moment), or `:c` (momentum with conditional second
moment). Mode `:c` uses the simple conditional estimator from Equation (26)
of Jiang and Xiao. Convenience wrappers `bcos_g`, `bcos_m`, and `bcos_c` are
also provided.
"""
function bcos(x0::AbstractArray{Float64}, simulator::Function,
		iters::AbstractVector, eta::Float64, beta::Float64, epsilon::Float64,
		mode::Symbol, atest::AbstractMatrix, ytest::AbstractArray,
		compute_accuracy::Function)
	eta > 0.0 || throw(ArgumentError("eta must be positive"))
	_check_probability(beta, "beta")
	epsilon > 0.0 || throw(ArgumentError("epsilon must be positive"))
	mode in (:g, :m, :c) ||
		throw(ArgumentError("mode must be :g, :m, or :c"))
	niters = _optimizer_iterations(iters)

	x = copy(x0)
	m = zeros(size(x0))
	v = zeros(size(x0))
	initialized = false
	beta_v = 1.0 - (1.0 - beta)^2
	fs = zeros(niters)
	acs = Float64[]

	for i in 1:niters
		f, g = simulator(x)
		fs[i] = f

		if !initialized
			m .= g
			v .= g .^ 2
			initialized = true
		end

		if mode === :c
			v .= beta_v .* (m .^ 2) .+ (1.0 - beta_v) .* (g .^ 2)
			m .= beta .* m .+ (1.0 - beta) .* g
			direction = m
		elseif mode === :m
			m .= beta .* m .+ (1.0 - beta) .* g
			direction = m
			v .= beta .* v .+ (1.0 - beta) .* (direction .^ 2)
		else
			direction = g
			v .= beta .* v .+ (1.0 - beta) .* (direction .^ 2)
		end

		x .-= eta .* direction ./ sqrt.(v .+ epsilon)
		push!(acs, compute_accuracy(x, ytest, atest))
	end

	return x, fs, acs
end

bcos_g(x0::AbstractArray{Float64}, simulator::Function,
	iters::AbstractVector, eta::Float64, beta::Float64, epsilon::Float64,
	atest::AbstractMatrix, ytest::AbstractArray, compute_accuracy::Function) =
	bcos(x0, simulator, iters, eta, beta, epsilon, :g,
		atest, ytest, compute_accuracy)

bcos_m(x0::AbstractArray{Float64}, simulator::Function,
	iters::AbstractVector, eta::Float64, beta::Float64, epsilon::Float64,
	atest::AbstractMatrix, ytest::AbstractArray, compute_accuracy::Function) =
	bcos(x0, simulator, iters, eta, beta, epsilon, :m,
		atest, ytest, compute_accuracy)

bcos_c(x0::AbstractArray{Float64}, simulator::Function,
	iters::AbstractVector, eta::Float64, beta::Float64, epsilon::Float64,
	atest::AbstractMatrix, ytest::AbstractArray, compute_accuracy::Function) =
	bcos(x0, simulator, iters, eta, beta, epsilon, :c,
		atest, ytest, compute_accuracy)

"""
		gradagrad(x0, simulator, iters, gamma0, rho, r,
		          atest, ytest, compute_accuracy)

Simplified scalar Grad-GradaGrad (GradaGrad). The recommended default is
`rho=2.0`; `r` limits each learning-rate increase and must be in `[0, 1]`.
"""
function gradagrad(x0::AbstractArray{Float64}, simulator::Function,
		iters::AbstractVector, gamma0::Float64, rho::Float64, r::Float64,
		atest::AbstractMatrix, ytest::AbstractArray,
		compute_accuracy::Function)
	gamma0 > 0.0 || throw(ArgumentError("gamma0 must be positive"))
	rho > 0.0 || throw(ArgumentError("rho must be positive"))
	0.0 <= r <= 1.0 || throw(ArgumentError("r must satisfy 0 <= r <= 1"))
	niters = _optimizer_iterations(iters)

	x = copy(x0)
	gamma = gamma0
	alpha = 0.0
	previous_gradient = zeros(size(x0))
	fs = zeros(niters)
	acs = Float64[]

	for i in 1:niters
		f, g = simulator(x)
		fs[i] = f
		v_k = sum(abs2, g) - rho * dot(vec(g), vec(previous_gradient))

		if v_k >= 0.0
			alpha += v_k
		else
			v_k = max(v_k, -r * alpha)
			gamma *= sqrt(1.0 - v_k / max(alpha, eps(Float64)))
		end

		step_size = gamma / sqrt(max(alpha, eps(Float64)))
		x .-= step_size .* g
		previous_gradient .= g
		push!(acs, compute_accuracy(x, ytest, atest))
	end

	return x, fs, acs
end

# Alias matching the abbreviated name used in some descriptions.
gradgrad(args...) = gradagrad(args...)


