
using LinearAlgebra
using Random

Random.seed!(123)

function read_dataset(file_path::String, columnsa::Int, nb_classes::Int)
	lines = filter(!isempty, strip.(readlines(file_path)))
	m = length(lines)


	max_feature_index = columnsa

	a = zeros(Float64, m, max_feature_index)
	y = zeros(Float64, m)
	classmap = Dict{Int, Int}()
	class_counter = 0.0

	aux_classes = []
	for line in lines
		push!(aux_classes, parse(Int, split(line)[1]))
	end
	first_class = minimum(aux_classes)
	last_class = maximum(aux_classes)
	println("Classes in dataset: ", first_class, " to ", last_class)
	i = 1
	for c in first_class:last_class
		classmap[c] = i
		i += 1
	end

	for (k, v) in classmap
		if v == 0
			println("Class ", k, " mapped to 0.0")
			readline()
		end
	end

	for (i, line) in enumerate(lines)
		tokens = split(line)

		y[i] = parse(Float64, tokens[1])

		for feature in tokens[2:end]
			index_str, value_str = split(feature, ":", limit = 2)
			a[i, parse(Int, index_str)] = parse(Float64, value_str)
		end
	end

	return a, y, classmap
end

function test_loss_libsvm(data_set_name::String = "a9a")
	dataset_path = joinpath(@__DIR__, "$(data_set_name).txt")
	a, y = read_dataset(dataset_path)
	println("Dataset loaded: ", size(a), " samples, ", size(a, 2), " features")
	println("First 5 labels: ", y[1:5])
	println("First 5 feature vectors: ", a[1:5, 1:5])
	loss = (x::AbstractVector) -> return l1_logistic_loss(x, y, a)
	grad = (x::AbstractVector) -> return l1_logistic_gradient(x, y, a)
	simulator = (x::AbstractVector) -> return (loss(x), grad(x))

	x = randn(size(a, 2)) .+ 0.5

end

# test_loss_libsvm()
