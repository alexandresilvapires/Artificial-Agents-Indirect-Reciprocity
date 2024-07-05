module utils

using Plots
using Memoization
using SparseArrays
using Dates
using ArnoldiMethod

# Add execution error to strategies
function add_exec_error(strat::Vector, errorExecut::Float32)::Vector
    return (1 - 2 * errorExecut) .* strat .+ errorExecut
end

# Add assignment errors to social norms
function add_sn_error(socialn::Int, errorAssign::Float32)
    return (1 - 2 * errorAssign) * socialn + errorAssign
end
function make_social_norm_error(socialnorm::Vector, errorAssign::Float32)::Vector
    return [add_sn_error(socialnorm[1], errorAssign), add_sn_error(socialnorm[2], errorAssign), add_sn_error(socialnorm[3], errorAssign), add_sn_error(socialnorm[4], errorAssign)]
end

# Strategies, where p(G,B) = (prob of cooperating against Good, " against Bad)
const allC::Vector = [1, 1]
const allD::Vector = [0, 0]
const disc::Vector = [1, 0]

function find_pos_of_state(states::Vector, state::Tuple{Int,Int,Int})::Int
    # Find the position without using the lookup table
    return findfirst(x -> x == state, states)
end

function create_lookup_table(states::Vector)::Dict{Tuple{Int, Int, Int}, Int}
    lookup_table = Dict{Tuple{Int, Int, Int}, Int}()

    for (index, state) in enumerate(states)
        lookup_table[state] = index
    end

    return lookup_table
end

function pos_of_state(table::Dict{Tuple{Int, Int, Int}, Int}, state::Tuple{Int, Int, Int})::Int
    get(table, state, -1)  
end

function get_transition_matrix_statdist(transition_matrix::SparseMatrixCSC)::Vector{Float32}
    transition_matrix_transp = transpose(transition_matrix)

    decomp,_ = partialschur(transition_matrix_transp, nev=1);
    stat_dist = vec(real(decomp.Q))
    stat_dist /= sum(stat_dist)
    return stat_dist
end


function make_plot_folder(foldername::String="")::String
    # Get current date and hour
    current_date_hour = Dates.now()

    # Create a formatted string for the folder name
    folder_name = (foldername == "" ? Dates.format(current_date_hour, "#Y#m#d_#H#M") : foldername)
    
    # Create the path for the folder
    folder_path = joinpath("Plots", folder_name)
    
    # Create the folder if it doesn't exist
    isdir(folder_path) || mkdir(folder_path)

    return folder_path
end

function write_parameters_txt(path::AbstractString, parameters)
    try
        open(path, "w") do file
            for (param, value) in parameters
                println(file, "$param = $value")
            end
        end
        println("Parameters written to $path\n")
    catch e
        println("Error writing parameters: $e\n")
    end
end

function write_result_txt(path::AbstractString, population, result)
    try
        open(path, "a") do file
            println(file, "$population -> $result")
        end
        #println("Data written to $path\n")
    catch e
        println("Error writing data: $e\n")
    end
end

function prepend_diff(v)
    newarray = Float32[0.0]
    for i in 2:lastindex(v)
        push!(newarray, v[i]-v[i-1])
    end
    return newarray
end

function read_parameters(file_path::AbstractString)
    params = Dict{AbstractString, Any}()

    open(file_path, "r") do file
        for line in eachline(file)
            parts = split(line, "=")
            key = strip(parts[1])
            value = try
                parse(Float64, strip(parts[2]))
            catch
                parse.(Float64, split(strip(parts[2]), ","))
            end
            params[key] = value
        end
    end

    return params
end

function resultstxt_to_plot(path::AbstractString, AAsPop::AbstractString)
    # Read data from the results.txt file
    data = Dict{AbstractString, Vector{Float32}}()

    results_path = joinpath(folder_path, "results.txt")
    parameters_path = joinpath(folder_path, "results.txt")

    open(results_path, "r") do file
        for line in eachline(file)
            parts = split(line, "->")
            key = strip(parts[1])
            values = parse.(Float32, split(strip(parts[2]), ","))
            data[key] = values
        end
    end

    params = read_parameters(parameters_path)
    fractions_AAs = get(params, "Fractions", [])

    # Create a single plot for the specified AAsPop
    p = plot(xlabel="Fraction of $AAsPop AAs", xticks=0:0.1:maximum(fractions_AAs), yticks=0:0.1:1, xlims=(0, maximum(fractions_AAs)), ylims=(0, 1),
            legend=:topright, frame=:true, grid=:true,
            ylabel="Cooperation Index", title="Cooperation Index for $AAsPop",
            size=(800, 500))

    line_colors = (RGB(0.3686, 0.5059, 0.7098), RGB(0.8784, 0.6118, 0.1412), RGB(0.5608, 0.6902, 0.1961), RGB(0.9216, 0.3843, 0.2078))
    labels = ["Image Score", "Simple-Standing", "Shunning", "Stern-Judging"]

    for (label, color) in zip(labels, line_colors)
        key = "$AAsPop:$label"
        values = get(data, key, nothing)
        if values !== nothing
            plot!(p, fractions_AAs, values_for_norm, label=label, linecolor=color,linewidth=1.2)
        else
            println("No data found for $key")
        end
    end

    return p
end


function processResults(filepath::String, population::String)::Vector{Vector{Float32}}
    result_arrays = Vector{Vector{Float32}}()
    open(joinpath(filepath, "results.txt")) do file
        for line in eachline(file)
            if startswith(line, population)
                # Extract the array part from the line
                array_part = split(line, " -> ")[2]
                # Remove "Float32[" and "]" from the array part
                array_string = replace(array_part, r"Float32\[|\]" => "")
                # Convert the comma-separated string into an array of Float32
                array = parse.(Float32, split(array_string, ", "))
                push!(result_arrays, array)
            end
        end
    end
    return result_arrays
end

function parse_float32_array(array_string::String)::Vector{Float32}
    # Remove "Float32[" and "]" from the array string
    array_string = replace(array_string, r"Float32\[|\]" => "")
    # Evaluate the string as Julia code
    array_expr = Meta.parse(array_string)
    # Convert the expression to a tuple of Float32
    array_tuple = eval(array_expr)
    # Convert the tuple to a vector
    array = collect(array_tuple)
    return array
end



function getParameterValue(filepath::String, parameter::String)
    value = ""
    open(joinpath(filepath, "parameters.txt")) do file
        for line in eachline(file)
            if startswith(line, parameter)
                # Extract the parameter value
                value = split(line, " = ")[2]
                break
            end
        end
    end
    return value
end


function generate_log_spaced_values(start_val::Float64, end_val::Float64, num_samples::Int)::Vector
    if start_val <= 0
        start_val = 1e-10  # Set a small positive value instead of 0
    end
    log_start = log10(start_val)
    log_end = log10(end_val)
    log_spaced_vals = Float32(10) .^ LinRange(log_start, log_end, num_samples)
    return log_spaced_vals
end

end
