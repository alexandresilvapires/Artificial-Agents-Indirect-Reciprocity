using Plots, Measures
using LaTeXStrings
using LinearAlgebra
using Memoization
using Dates
using SparseArrays
using ArnoldiMethod
using Base.Threads

include("./utils.jl")
include("./strategy_dynamics.jl")

const foldername = "example_b_study"

const popsize::Int = 100  # Size of the population
const pop_sample::Int = 25 # Size of population for reputation dynamics

const fixedAAreputations::Bool = false      # If true, AAs have fixed reputations
const fixedAARep::String = "G"              # The fixed reputations the AAs have
const canImitateAAs = true                  # If true, adaptive agents can imitate AAs

const errorAssign::Float32 = 0.01    # Prob. of assigning the wrong reputation after an observation
const errorExecut::Float32 = 0.01    # Prob. of executing the wrong action
const errorAssess::Float32 = 0.01    # Prob. of incorrectly accessing the other's reputation before choosing an action

const strengthOfSelection::Float32 = 1
const mutationChance::Float32 = 1.0 / popsize

const bs::Vector{Float32} = [1+i for i in 0:0.25:7] # benefit of cooperation values to try. c is always 1.0

# Fractions of AAs to be tested
const fractions_AAs::Vector{Float32}  = [0]
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
        "Type" => "Studying coop for different bs",
        "errorAssign" => errorAssign,
        "errorExecut" => errorExecut,
        "errorAssess" => errorAssess,
        "strengthOfSelection" => strengthOfSelection,
        "mutationChance" => mutationChance,
        "Approximation Method" => "Average Rep State with Downsampling",
        "Population Sample Size" => pop_sample
    )
    utils.write_parameters_txt(joinpath(folder_path, "parameters.txt"),parameters)
end

function plot_b_study() 
    folder_path = utils.make_plot_folder(foldername)

    write_parameters(folder_path)

    values_for_norm = zeros(Float32, length(vectorNAAs))

    line_colors = (RGB(0.3686, 0.5059, 0.7098), RGB(0.8784, 0.6118, 0.1412), RGB(0.5608,0.6902,0.1961), RGB(0.9216, 0.3843, 0.2078),RGB(0.5, 0.5, 0.5))
    markerTypes = [:circle, :square, :dtriangle, :utriangle, :diamond]

    # Create a single plot for each AAsPop
    p = plot(xlabel="Donation benefit, b", xticks=1:1:maximum(8), yticks=0:0.1:1, xlims=(1, maximum(bs)+0.01), ylims=(0, 1),
        legend=:outertopright, frame=:true, grid=:true,
        ylabel=L"Cooperation Index, $\mathit{I}$",
        size=(800, 500), dpi=1000, tickfont=font(10),xguidefontsize=20, yguidefontsize=20)  

    values_for_norm = zeros(Float32, length(bs))    

    for (norm, color, label, markerType) in zip(norms, line_colors, ["Image Score", "Simple-Standing", "Shunning", "Stern-Judging", "All Good"],markerTypes)
        Threads.@threads for i in eachindex(bs)
            print("\nrunning args sn=",label," b=",  bs[i])
            values_for_norm[i] = coop_index(norm, 0, 0, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, bs[i], mutationChance, strengthOfSelection)
        end

        Memoization.empty_all_caches!()

        # Add lines incrementally for each norm
        plot!(p, bs, values_for_norm, label=label, linecolor=color,legendfontsize=12,linewidth=2,marker=(markerType,3),markercolor=color,markerstrokewidth=0)

        utils.write_result_txt(joinpath(folder_path, "results.txt"),label,values_for_norm)
    end 

    # Save the plot inside the folder
    plot_path = joinpath(folder_path, "b_study.png")
    savefig(p, plot_path)   

    Memoization.empty_all_caches!()
end

@time begin
    plot_b_study() 
end
