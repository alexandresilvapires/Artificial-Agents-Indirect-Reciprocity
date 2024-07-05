using Plots
using LinearAlgebra
using Memoization
using Dates
using SparseArrays
using ArnoldiMethod
using Base.Threads

include("./utils.jl")
include("./strategy_dynamics.jl")

const foldername = "example_coop_study"

const popsize::Int = 100  # Size of the population
const pop_sample::Int = 25 # Size of population for reputation dynamics

const fixedAAreputations::Bool = false      # If true, AAs have fixed reputations
const fixedAARep::String = "G"              # The fixed reputations the AAs have
const canImitateAAs = true                  # If true, adaptive agents can imitate AAs

const errorAssign::Float32 = 0.01    # Prob. of assigning the wrong reputation after an observation
const errorExecut::Float32 = 0.01    # Prob. of executing the wrong action
const errorAssess::Float32 = 0.01    # Prob. of incorrectly accessing the other's reputation before choosing an action

const strengthOfSelection::Float32 = 1.0
const mutationChance::Float32 = 1.0 / popsize

const b::Float32 = 2  # benefit of cooperation. c is always 1.0

# Fractions of AAs to be tested
const fractions_AAs::Vector{Float32}  = [0; [Float32(i / pop_sample) for i in 1:(pop_sample-1)]]
const vectorNAAs::Vector{Int} = Int.(round.(popsize .* fractions_AAs))

# Strategies of AAs to test. Add to vector to test more. Eg: ["AllC","Disc"] tests first AllC AAs, then Disc AAs 
const AA_pops_to_calc::Vector{String} = ["Disc"]

# Social norms to test
const norm_IS::Vector = utils.make_social_norm_error([1, 0, 1, 0], errorAssign)
const norm_SS::Vector = utils.make_social_norm_error([1, 0, 1, 1], errorAssign)
const norm_SH::Vector = utils.make_social_norm_error([1, 0, 0, 0], errorAssign)
const norm_SJ::Vector = utils.make_social_norm_error([1, 0, 0, 1], errorAssign)
const norm_AG::Vector = utils.make_social_norm_error([1, 1, 1, 1], errorAssign)
const norms::Vector = [norm_IS, norm_SS, norm_SH, norm_SJ, norm_AG]

Memoization.empty_all_caches!()

function write_parameters(folder_path::String)
    parameters = (
        "Population Size" => popsize,
        "fixedAAreputations" => fixedAAreputations,
        "fixedAARep" => fixedAARep,
        "Imit AAs" => canImitateAAs,
        "errorAssign" => errorAssign,
        "errorExecut" => errorExecut,
        "errorAssess" => errorAssess,
        "strengthOfSelection" => strengthOfSelection,
        "mutationChance" => mutationChance,
        "b" => b,
        "Fractions" => fractions_AAs,
        "Approximation Method" => "Average Rep State with Downsampling",
        "Population Sample Size" => pop_sample
    )
    utils.write_parameters_txt(joinpath(folder_path, "parameters.txt"),parameters)
end

function plot_coop_index() 
    folder_path = utils.make_plot_folder(foldername)

    write_parameters(folder_path)

    values_for_norm = zeros(Float32, length(vectorNAAs))

    line_colors = (RGB(0.3686, 0.5059, 0.7098), RGB(0.8784, 0.6118, 0.1412), RGB(0.5608,0.6902,0.1961), RGB(0.9216, 0.3843, 0.2078),RGB(0.5, 0.5, 0.5))
    for AAsPop in AA_pops_to_calc
        index_AA = AAsPop == "AllC" ? 0 : (AAsPop == "AllD" ? 1 : 2)

        # Create a single plot for each AAsPop
        p = plot(xlabel="Fraction of $AAsPop AAs", xticks=0:0.1:maximum(fractions_AAs), yticks=0:0.1:1, xlims=(0, maximum(fractions_AAs)), ylims=(0, 1),
            legend=:outertopright, frame=:true, grid=:true,
            ylabel="Cooperation Index", title="Cooperation Index for $AAsPop",
            size=(800, 500), dpi=1000)

        values_for_norm = zeros(Float32, length(vectorNAAs))

        for (norm, color, label) in zip(norms, line_colors, ["Image Score", "Simple-Standing", "Shunning", "Stern-Judging", "All Good"])
            Threads.@threads for i in eachindex(vectorNAAs)
                print("\nRunning sn=",label," nAAs=",  vectorNAAs[i]," index=",  index_AA," ")
                values_for_norm[i] = coop_index(norm, vectorNAAs[i], index_AA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, strengthOfSelection)
            end
            
            Memoization.empty_all_caches!()

            # Add lines incrementally for each norm
            plot!(p, fractions_AAs, values_for_norm, label=label, linecolor=color,linewidth=1.2)
            
            utils.write_result_txt(joinpath(folder_path, "results.txt"),AAsPop*":"*label,values_for_norm)
        end


        # Save the plot inside the folder
        plot_path = joinpath(folder_path, "#$AAsPop"*"_coop_index.png")
        savefig(p, plot_path)

        Memoization.empty_all_caches!()
    end
end

@time begin
    plot_coop_index()
end
