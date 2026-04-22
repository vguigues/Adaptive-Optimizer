
using LinearAlgebra
using Random

Random.seed!(123)

function read_dataset(file_path::String,columnsa::Int)
    lines = filter(!isempty, strip.(readlines(file_path)))
    m = length(lines)


    max_feature_index = columnsa

    a = zeros(Float64, m, max_feature_index)
    y = zeros(Float64, m)

    for (i, line) in enumerate(lines)
        tokens = split(line)
        y[i] = parse(Float64, tokens[1])

        for feature in tokens[2:end]
            index_str, value_str = split(feature, ":", limit=2)
            a[i, parse(Int, index_str)] = parse(Float64, value_str)
        end
    end

    return a, y
end

function test_loss_libsvm(data_set_name::String="a9a")
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
