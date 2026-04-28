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


