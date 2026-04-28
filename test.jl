
using LinearAlgebra
using Random
using Plots
include("adaptive_optimizer.jl")
include("readdataset.jl")

Random.seed!(123)

function multiclass_accuracy(x::AbstractMatrix{T}, y::AbstractMatrix{T}, a::AbstractMatrix{T}) where {T <: Real}
	m, n = size(a)
	c = size(y, 2)
	probs = softmax(a * x)

	correct = 0
	for i in 1:m
		pred_class = argmax(probs[i, :])
		true_class = argmax(y[i, :])
		correct += (pred_class == true_class)
	end

	return correct / m
end

function l1_logistic_loss(x::AbstractVector{T}, y::AbstractVector{T}, a::AbstractMatrix{T}) where {T <: Real}
	m = size(a, 1)
	loss = 0.0
	epsilon = 1e-4
	norm_x = norm(x, 1)
	for i in 1:m
		loss += log(1 + exp(-y[i] * dot(a[i, :], x)))
	end
	return (loss / m) + epsilon*norm_x
end


function l1_logistic_gradient(x::AbstractVector{T}, y::AbstractVector{T}, a::AbstractMatrix{T}) where {T <: Real}
	n = length(x)
	m = size(a, 1)
	gradient = zeros(T, n)
	epsilon = 1e-4
	for i in 1:m
		exp_term = exp(-y[i] * dot(a[i, :], x))
		gradient-=(y[i]*a[i, :]'*exp_term/(1+exp_term))'
	end
	return (gradient / m) + epsilon * sign.(x)
end


function softmax(z::AbstractMatrix{T}) where {T <: Real}
	z_shifted = z .- maximum(z, dims = 2)
	exp_z = exp.(z_shifted)
	return exp_z ./ sum(exp_z, dims = 2)
end


# exp(z[i,j])/sum(exp(z[i,k],k=1:c)) 


function cross_entropy_loss(x::AbstractMatrix{T}, y::AbstractMatrix{T}, a::AbstractMatrix{T}) where {T <: Real}
	m, n = size(a)
	c = size(y, 2)
	@assert size(y, 1) == m "y must have m rows (same number of samples as a)."
	@assert size(x, 1) == n "x must have n rows, where n=size(a,2)."
	@assert size(x, 2) == c "x must have c columns, where c=size(y,2)."

	logits = a * x
	probs = softmax(logits)
	eps = T(1e-12)

	return -sum(y .* log.(probs .+ eps)) / m
end


function cross_entropy_gradient(x::AbstractMatrix{T}, y::AbstractMatrix{T}, a::AbstractMatrix{T}) where {T <: Real}
	m, n = size(a)
	c = size(y, 2)
	@assert size(y, 1) == m "y must have m rows (same number of samples as a)."
	@assert size(x, 1) == n "x must have n rows, where n=size(a,2)."
	@assert size(x, 2) == c "x must have c columns, where c=size(y,2)."

	logits = a * x
	probs = softmax(logits)

	return (a' * (probs - y)) / m
end


function simulator_logistic(x::AbstractVector{T}, y::AbstractVector{T}, a::AbstractMatrix{T}) where {T <: Real}
	f=l1_logistic_loss(x, y, a)
	gradient=l1_logistic_gradient(x, y, a)
	return f, gradient
end

function simulator_cross_entropy(x::AbstractMatrix{T}, y::AbstractMatrix{T}, a::AbstractMatrix{T}) where {T <: Real}
	f=cross_entropy_loss(x, y, a)
	gradient=cross_entropy_gradient(x, y, a)
	return f, gradient
end


function svm_loss(training_data_set_path::String, testing_data_set_path::String, iter::Int, eta::Float64, epsilon::Float64, columnsa::Int, D::Float64, gamma::Float64, beta::Float64, beta1::Float64, beta2::Float64, alpha::Float64, beta3::Float64)

	a, y = read_dataset(training_data_set_path, columnsa)
	# loss = (x::AbstractVector)  -> return l1_logistic_loss(x, y, a) 
	# grad = (x::AbstractVector)  -> return l1_logistic_gradient(x, y, a)
	simulator=(x::AbstractVector) -> return simulator_logistic(x, y, a)

	x=zeros(size(a, 2))
	x1, fs1=adaptive_optimizer(x, simulator, iter, D)
	x=zeros(size(a, 2))
	x2, fs2=adaptive_optimizer_cwise(x, simulator, iter, D, epsilon)
	x=zeros(size(a, 2))
	x3, fs3=adagrad(x, simulator, iter, eta, epsilon)
	x=zeros(size(a, 2))
	x4, fs4=ema(x, simulator, iter, beta, gamma, epsilon, beta3)
	x=zeros(size(a, 2))
	x5, fs5=emawise(x, simulator, iter, beta, gamma, epsilon, beta3)
	x=zeros(size(a, 2))
	x6, fs6=adam(x, simulator, iter, beta1, beta2, epsilon, alpha)

	atest, ytest = read_dataset(testing_data_set_path, columnsa)

	accuracy1 = sum(sign.(atest * x1) .== ytest) / length(ytest)
	accuracy2 = sum(sign.(atest * x2) .== ytest) / length(ytest)
	accuracy3 = sum(sign.(atest * x3) .== ytest) / length(ytest)
	accuracy4 = sum(sign.(atest * x4) .== ytest) / length(ytest)
	accuracy5 = sum(sign.(atest * x5) .== ytest) / length(ytest)
	accuracy6 = sum(sign.(atest * x6) .== ytest) / length(ytest)

	return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6
end

function one_hot(y::Vector, K::Int, classmap::Dict{Int, Int})
	m = length(y)
	Y = zeros(Float64, m, K)
	for i in 1:m
		Y[i, classmap[Int(y[i])]] = 1.0
	end
	return Y
end

function test_svm_loss(training_data_set_path::String, testing_data_set_path::String, epsilon::Float64, columnsa::Int, nbClass::Int)

	# iters=Vector(1:1:100)
	iters=Vector(1:1:50)

	# Adaptive optimizer parameters
	Ds=[10^(-3), 10^(-2), 10^(-1), 1.0, 10.0]

	# Adagrad parameters
	etas=[10^(-3), 10^(-2), 10^(-1), 1.0, 10.0]

	# EMA parameters
	gammas=[10^(-3), 10^(-2), 10^(-1), 1.0, 10.0]
	betas=[0.9, 0.96, 0.98]
	beta3s=[0.9, 0.96, 0.98]

	# ADAM parameters
	beta1s=[0.9, 0.99, 0.999]
	beta2s=[0.9, 0.99, 0.999]
	alphas=[10^(-3), 10^(-2), 10^(-1), 1.0, 10.0]

	max_ac1 = 0
	max_ac2 = 0
	max_ac3 = 0
	max_ac4 = 0
	max_ac5 = 0
	max_ac6 = 0

	# Best adaptive optimizer parameters
	best_D_adap=0

	# Best adaptive optimizer coordinate-wise parameters
	best_D_adapwise=0

	# Best adagrad parameters
	best_eta_adagrad=0

	# Best EMA parameters
	best_emagamma=0
	best_emabeta=0
	best_emabeta3=0

	# Best EMA coordinate-wise parameters
	best_emawisegamma=0
	best_emawisebeta=0
	best_emawisebeta3=0

	# Best ADAM parameters
	best_beta1=0
	best_beta2=0
	best_alpha=0

	# Adaptove optimizer objective function values
	best_fs1=[]
	# Adaptive optimizer coordinate-wise objective function values
	best_fs2=[]
	# Adagrad objective function values
	best_fs3=[]
	# EMA objective function values
	best_fs4=[]
	# EMA coordinate-wise objective function values
	best_fs5=[]
	# ADAM objective function values
	best_fs6=[]

	# Best accuracy adaptive optimizer
	max_ac1s = -1
	# Best accuracy adaptive optimizer coordinate-wise
	max_ac2s = -1
	# Best accuracy adagrad
	max_ac3s = -1
	# Best accuracy EMA
	max_ac4s = -1
	# Best accuracy EMA coordinate-wise
	max_ac5s = -1
	# Best accuracy ADAM
	max_ac6s = -1

	# Read the training dataset


	a, y, classmap = read_dataset(training_data_set_path, columnsa, nbClass)
	atest, ytest, _ = read_dataset(testing_data_set_path, columnsa, nbClass)

	print("Testing dataset loaded.")
	simulator=(x::AbstractVector) -> return simulator_logistic(x, y, a)
	x0=zeros(size(a, 2))

	compute_accuracy = (x::AbstractVector, y::AbstractVector, a::AbstractMatrix) -> return sum(sign.(a * x) .== y) / length(y)


	if nbClass > 2
		x0=zeros(size(a, 2), nbClass)
		y = one_hot(y, nbClass, classmap)
		ytest = one_hot(ytest, nbClass, classmap)
		# Define the simulator function for the cross-entropy loss
		simulator=(x::AbstractMatrix) -> return simulator_cross_entropy(x, y, a)
		compute_accuracy = (x::AbstractMatrix, y::AbstractMatrix, a::AbstractMatrix) -> return multiclass_accuracy(x, y, a)
	end

	# Find best values of the parameters and compute the objective along iterations for the adaptive optimizer
	for (j, D) in enumerate(Ds)
		x=copy(x0)
		x1, fs1, ac1s=adaptive_optimizer(x, simulator, iters, D, atest, ytest, compute_accuracy)
		if maximum(ac1s) > max_ac1
			max_ac1 = maximum(ac1s)
			best_D_adap = D
			max_ac1s = ac1s
			best_fs1=fs1
		end
	end

	# Find best values of the parameters and compute the objective along iterations for the adaptive optimizer coordinate-wise
	for (j, D) in enumerate(Ds)
		x=copy(x0)
		x2, fs2, ac2s=adaptive_optimizer_cwise(x, simulator, iters, D, epsilon, atest, ytest, compute_accuracy)
		if maximum(ac2s) > max_ac2
			max_ac2 = maximum(ac2s)
			best_D_adapwise = D
			max_ac2s = ac2s
			best_fs2=fs2
		end
	end

	# Find best values of the parameters and compute the objective along iterations for adagrad
	for (j, eta) in enumerate(etas)
		x=copy(x0)
		x3, fs3, ac3s=adagrad(x, simulator, iters, eta, epsilon, atest, ytest, compute_accuracy)
		if maximum(ac3s) > max_ac3
			max_ac3 = maximum(ac3s)
			best_eta_adagrad = eta
			max_ac3s = ac3s
			best_fs3=fs3
		end
	end

	# Find best values of the parameters and compute the objective along iterations for EMA
	for (j, gamma) in enumerate(gammas)
		for (k, beta) in enumerate(betas)
			for (l, beta3) in enumerate(beta3s)
				x=copy(x0)
				x4, fs4, ac4s=ema(x, simulator, iters, beta, gamma, epsilon, beta3, atest, ytest, compute_accuracy)
				if maximum(ac4s) > max_ac4
					max_ac4 = maximum(ac4s)
					best_emagamma=gamma
					best_emabeta=beta
					best_emabeta3=beta3
					max_ac4s = ac4s
					best_fs4=fs4
				end
			end
		end
	end

	# Find best values of the parameters and compute the objective along iterations for EMA coordinate-wise 
	for (j, gamma) in enumerate(gammas)
		for (k, beta) in enumerate(betas)
			for (l, beta3) in enumerate(beta3s)
				x=copy(x0)
				x5, fs5, ac5s=emawise(x, simulator, iters, beta, gamma, epsilon, beta3, atest, ytest, compute_accuracy)
				if maximum(ac5s) > max_ac5
					max_ac5 = maximum(ac5s)
					best_emawisegamma=gamma
					best_emawisebeta=beta
					best_emawisebeta3=beta3
					max_ac5s = ac5s
					best_fs5=fs5
				end
			end
		end
	end

	# Find best values of the parameters and compute the objective along iterations for ADAM
	for (j, beta1) in enumerate(beta1s)
		for (k, beta2) in enumerate(beta2s)
			for (l, alpha) in enumerate(alphas)
				x=copy(x0)
				x6, fs6, ac6s=adam(x, simulator, iters, beta1, beta2, epsilon, alpha, atest, ytest, compute_accuracy)

				if maximum(ac6s) > max_ac6
					max_ac6 = maximum(ac6s)
					best_beta1=beta1
					best_beta2=beta2
					best_alpha=alpha
					max_ac6s = ac6s
					best_fs6=fs6
				end
			end
		end
	end

	# Plot the accuracy
	name = splitext(basename(training_data_set_path))[1]
	abscissa = iters
	p = plot(abscissa, max_ac1s, label = "Flex optimizer", xlabel = "Iterations", ylabel = "Accuracy", title = name, linestyle = :solid)
	plot!(p, abscissa, max_ac2s, label = "Flex optimizer coordinate-wise", linestyle = :dash)
	plot!(p, abscissa, max_ac3s, label = "Adagrad", linestyle = :dash)
	plot!(p, abscissa, max_ac4s, label = "EMA", linestyle = :dash)
	plot!(p, abscissa, max_ac5s, label = "EMA coordinate-wise", linestyle = :dash)
	plot!(p, abscissa, max_ac6s, label = "ADAM", linestyle = :dash)
	display(p)
	@show name
	savefig(p, "$(name).pdf")

	abscissa = iters
	p = plot(abscissa, best_fs1, label = "Flex optimizer", xlabel = "Iterations", ylabel = "Objective function", title = name, linestyle = :solid)
	plot!(p, abscissa, best_fs2, label = "Flex optimizer coordinate-wise", linestyle = :dash)
	plot!(p, abscissa, best_fs3, label = "Adagrad", linestyle = :dash)
	plot!(p, abscissa, best_fs4, label = "EMA", linestyle = :dash)
	plot!(p, abscissa, best_fs5, label = "EMA coordinate-wise", linestyle = :dash)
	plot!(p, abscissa, best_fs6, label = "ADAM", linestyle = :dash)
	display(p)
	name = splitext(basename(training_data_set_path))[1]
	@show name
	savefig(p, "Objective_$(name).pdf")

end

# test_svm_loss("Data_SVM/a1a.txt", "Data_SVM_Testing/a1a.t", 10^(-5), 123, 2)
# test_svm_loss("Data_SVM/a2a.txt", "Data_SVM_Testing/a2a.t", 10^(-5), 123, 2)

# test_svm_loss("Data_SVM/sensorless.txt", "Data_SVM_Testing/sensorless.t", 10^(-5), 48, 11)
test_svm_loss("Data_SVM/aloi.txt", "Data_SVM_Testing/aloi.t", 10^(-5), 128, 1000)
# test_svm_loss("Data_SVM/dna.txt", "Data_SVM_Testing/dna.t", 10^(-5), 180,3)
# test_svm_loss("Data_SVM/glass.txt", "Data_SVM_Testing/glass.t", 10^(-5), 9,6)
# test_svm_loss("Data_SVM/iris.txt", "Data_SVM_Testing/iris.t", 10^(-5), 4,3)
# test_svm_loss("Data_SVM/letter.txt", "Data_SVM_Testing/letter.t", 10^(-5), 16,26)
# test_svm_loss("Data_SVM/pendigits.txt", "Data_SVM_Testing/pendigits.t", 10^(-5), 16,10)

# test_svm_loss("Data_SVM/a3a.txt", "Data_SVM_Testing/a3a.t", 10^(-5), 123, 2)
# test_svm_loss("Data_SVM/a4a.txt", "Data_SVM_Testing/a4a.t", 10^(-5), 123, 2)
# test_svm_loss("Data_SVM/a5a.txt", "Data_SVM_Testing/a5a.t", 10^(-5), 123, 2)
# test_svm_loss("Data_SVM/a6a.txt", "Data_SVM_Testing/a6a.t", 10^(-5), 123, 2)
# test_svm_loss("Data_SVM/a7a.txt", "Data_SVM_Testing/a7a.t", 10^(-5), 123, 2)
# test_svm_loss("Data_SVM/a8a.txt", "Data_SVM_Testing/a8a.t", 10^(-5), 123, 2)
# test_svm_loss("Data_SVM/a9a.txt", "Data_SVM_Testing/a9a.t", 10^(-5), 123, 2)
# test_svm_loss("Data_SVM/australian.txt","Data_SVM_Testing/australian.t",1000,0.01,10^(-5))
# test_svm_loss("Data_SVM/breast-cancer.txt","Data_SVM_Testing/breast-cancer.t",1000,0.01,10^(-5))
# test_svm_loss("Data_SVM/cod-ma.txt","Data_SVM_Testing/cod-ma.t",1000,0.01,10^(-5))
# test_svm_loss("Data_SVM/diabetes.txt","Data_SVM_Testing/diabetes.t",1000,0.01,10^(-5))
# test_svm_loss("Data_SVM/fourclass.txt","Data_SVM_Testing/fourclass.t",1000,0.01,10^(-5))
# test_svm_loss("Data_SVM/ionosphere_scale.txt","Data_SVM_Testing/ionosphere_scale.t",1000,0.01,1₀^(-5))
# test_svm_loss("Data_SVM/phishing.txt","Data_SVM_Testing/phishing.t",1000,0.01,10^(-5))
# test_svm_loss("Data_SVM/sonar_scale.txt","Data_SVM_Testing/sonar_scale.t",1000,0.01,10^(-5))
# test_svm_loss("Data_SVM/w1a.txt","Data_SVM_Testing/w1a.t",1000,0.01,10^(-5))
# test_svm_loss("Data_SVM/a9mushroomsa.txt","Data_SVM_Testing/a9mushroomsa.t",1000,0.01,10^(-5))






