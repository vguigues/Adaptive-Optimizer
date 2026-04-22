
using LinearAlgebra


function adaptive_optimizer(x0::AbstractVector{Float64}, simulator::Function, iters::AbstractVector, D::Float64, atest::AbstractMatrix, ytest::AbstractVector)
	n=size(x0)
	x=copy(x0)
	s=zeros(n);
	ssq=0
	fs=zeros(iters[end])
	xsum=zeros(n)
	acs=[]

	for i in 1:iters[end]
		f, g=simulator(x)
		fs[i]=f
		s+=g
		ssq+=norm(g, 2)^2
		gamma=D*norm(s, 2)/ssq
		x=x0-gamma*s
		ac1 = sum(sign.(atest * x) .== ytest) / length(ytest)
		push!(acs, ac1)
		xsum+=x
	end
	x.=xsum/iters[end]
	return x, fs, acs
end

function adaptive_optimizer_cwise(x0::AbstractVector{Float64}, simulator::Function, iters::AbstractVector, D::Float64, epsilon::Float64, atest::AbstractMatrix, ytest::AbstractVector)
	n=size(x0, 1)
	x=copy(x0)
	s=zeros(n)
	ssq=zeros(n)
	gamma=zeros(n)
	fs=zeros(iters[end])
	xsum=zeros(n)
	acs=[]
	for i in 1:iters[end]
		f, g=simulator(x)
		fs[i]=f
		s+=g
		for j in 1:n
			ssq[j]+=g[j]^2
			gamma[j]=D*abs(s[j])/(ssq[j]+epsilon)
			x[j]=x0[j]-gamma[j]*s[j]
		end
		xsum+=x
		ac1 = sum(sign.(atest * x) .== ytest) / length(ytest)
		# println("Adaptive optimizer iteration $i, accuracy: $ac1")
		push!(acs, ac1)
	end
	x.=xsum/iters[end]

	return x, fs, acs
end

function adagrad(x0::AbstractVector{Float64}, simulator::Function, iters::AbstractVector, eta::Float64, epsada::Float64, atest::AbstractMatrix, ytest::AbstractVector)
	n=size(x0)
	x=copy(x0)
	ssq=zeros(n)
	fs=zeros(iters[end])
	acs=[]

	for i in 1:iters[end]
		f, g=simulator(x)
		fs[i]=f
		ssq+=g .^ 2
		x=x-eta*g ./ sqrt.(epsada .+ ssq)
		ac1 = sum(sign.(atest * x) .== ytest) / length(ytest)
		# println("Adagrad iteration $i, accuracy: $ac1")
		push!(acs, ac1)
	end
	return x, fs, acs
end

function ema(x0::AbstractVector{Float64}, simulator::Function, iters::AbstractVector, beta::Float64, gamma::Float64, epsilon::Float64, beta3::Float64, atest::AbstractMatrix, ytest::AbstractVector)
	x=copy(x0)
	n=size(x0)
	s=zeros(n)
	m=zeros(n)
	fs=zeros(iters[end])
	v=0
	acs=[]
	for i in 1:iters[end]
		f, g=simulator(x)
		fs[i]=f
		s=beta*s+(1-beta)*g
		v=beta*v+(1-beta)*norm(g, 2)^2
		m=beta3*m+(1-beta3)*g
		x-=gamma*(norm(s, 2)/(v+epsilon))*m
		ac1 = sum(sign.(atest * x) .== ytest) / length(ytest)
		push!(acs, ac1)
	end
	return x, fs, acs
end

function emawise(x0::AbstractVector{Float64}, simulator::Function, iters::AbstractVector, beta::Float64, gamma::Float64, epsilon::Float64, beta3::Float64, atest::AbstractMatrix, ytest::AbstractVector)
	x=x0
	n=size(x0)
	s=zeros(n)
	fs=zeros(iters[end])
	v=zeros(n)
	m=zeros(n)
	acs=[]
	for i in 1:iters[end]
		f, g=simulator(x)
		fs[i]=f
		s=beta*s+(1-beta)*g
		m=beta3*m+(1-beta3)*g
		v=beta*v+(1-beta)*(g .^ 2)
		x-=gamma*((abs.(s) ./ (v .+ epsilon)) .* m)
		ac1 = sum(sign.(atest * x) .== ytest) / length(ytest)
		push!(acs, ac1)
	end
	return x, fs, acs
end

function adam(x0::AbstractVector{Float64}, simulator::Function, iters::AbstractVector, beta1::Float64, beta2::Float64, epsilon::Float64, alpha::Float64, atest::AbstractMatrix, ytest::AbstractVector)
	x=x0
	n=size(x0)
	m=zeros(n)
	v=zeros(n)
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
		ac1 = sum(sign.(atest * x) .== ytest) / length(ytest)
		push!(acs, ac1)
	end
	return x, fs, acs
end


