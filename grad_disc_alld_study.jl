using Plots, Measures
using LinearAlgebra
using Memoization
using Dates
using SparseArrays
using ArnoldiMethod
using Base.Threads

include("./utils.jl")
include("./strategy_dynamics.jl")

const foldername = "example_grad_study"

const popsize::Int = 100  # Size of the population
const pop_sample::Int = 25 # Size of population for reputation dynamics

const fixedAAreputations::Bool = false      # If true, AAs have fixed reputations
const fixedAARep::String = "G"              # The fixed reputations the AAs have
const canImitateAAs = false                  # If true, adaptive agents can imitate AAs

const errorAssign::Float32 = 0.01    # Prob. of assigning the wrong reputation after an observation
const errorExecut::Float32 = 0.01    # Prob. of executing the wrong action
const errorAssess::Float32 = 0.01    # Prob. of incorrectly accessing the other's reputation before choosing an action

const strengthOfSelection::Float32 = 1.0
const mutationChance::Float32 = 1.0 / popsize

const b::Float32 = 2.0  # benefit of cooperation. c is always 1.0

# Fractions of AAs to be tested
const fractions_AAs::Vector{Float32} = [0, 1/pop_sample, 2/pop_sample, 3/pop_sample]
const vectorNAAs::Vector{Int} = Int.(round.(popsize .* fractions_AAs))

# Strategies of AAs to test. Add to vector to test more. Eg: ["AllC","Disc"] tests first AllC AAs, then Disc AAs 
const AA_pops_to_calc::Vector{String} = ["Disc"]

# Social norms to test
const norm_IS::Vector = utils.make_social_norm_error([1, 0, 1, 0], errorAssign)
const norm_SS::Vector = utils.make_social_norm_error([1, 0, 1, 1], errorAssign)
const norm_SH::Vector = utils.make_social_norm_error([1, 0, 0, 0], errorAssign)
const norm_SJ::Vector = utils.make_social_norm_error([1, 0, 0, 1], errorAssign)
const norm_AG::Vector = utils.make_social_norm_error([1, 1, 1, 1], errorAssign)
const norms::Vector = [norm_IS, norm_SS, norm_SH, norm_SJ]

Memoization.empty_all_caches!()

function write_parameters(folder_path::String)
    parameters = (
        "Population Size" => popsize,
        "fixedAAreputations" => fixedAAreputations,
        "fixedAARep" => fixedAARep,
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

function plot_grad_study() 
    folder_path = utils.make_plot_folder(foldername)

    write_parameters(folder_path)

    line_colors = (RGB(0.3686, 0.5059, 0.7098), RGB(0.8784, 0.6118, 0.1412), RGB(0.5608,0.6902,0.1961), RGB(0.9216, 0.3843, 0.2078),RGB(0.5, 0.5, 0.5))

    l = @layout [a b c d]
    norm_names = ["Image Score", "Simple-Standing", "Shunning", "Stern-Judging", "All Good"]

    for AAsPop in AA_pops_to_calc
        index_AA = AAsPop == "AllC" ? 0 : (AAsPop == "AllD" ? 1 : 2)

        subplots = []

        maxRes = -Inf
        minRes = Inf
        for i in eachindex(norms)

            ylab = (i == 1 ? "Gradient of Selection" : "")

            leg = (i == length(norms) ? :bottomleft : false)
            # Create a single plot for each norm
            p = plot(xlabel="Fraction of Disc",
                legend=leg, frame=:true, grid=:true,
                ylabel=ylab, title=norm_names[i])
            
            # create each line and add it to the plot
            for nAAs in vectorNAAs

                rr = 0:Int(popsize/pop_sample):(popsize-nAAs)
                results = [grad_disc_alld(0, nAllD, norms[i], nAAs, index_AA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, strengthOfSelection) for nAllD in rr]

                maxRes,minRes = max(maximum(results),maxRes), min(minimum(results), minRes)

                rrFrac = [k/popsize for k in rr]
                plot!(p, rrFrac, results, label=false, linecolor=line_colors[i],linewidth=2, linealpha=(1-(nAAs*6.5)/popsize))

                # Check for intersections with x-axis
                for t in 1:length(results)-1
                    if sign(results[t]) * sign(results[t+1]) < 0
                        # Interpolate to find the precise intersection point
                        m = (results[t+1] - results[t]) / (rrFrac[t+1] - rrFrac[t])
                        c = results[t] - m * rrFrac[t]
                        x_intersection = -c / m
                        scatter!([x_intersection], [0], markercolor=line_colors[i], markeralpha=(1-(nAAs*6.5)/popsize), markersize=4, label=false, markerstrokewidth=0)
                    end
                end
            end

            # plot a black line at 0
            plot!(p, 0:1:100, [0 for k in 0:1:100], linecolor=:black,linewidth=1.5, linealpha=1, label=false)
            plot!(p,xticks=0:0.1:1, yticks=-1:0.01:1)

            push!(subplots, p)
        end

        # last plot has fake lines to put label in gray
        for nAAs in vectorNAAs
            rr = 0:Int(popsize/pop_sample):(popsize-nAAs)
            results = [-1000 for nAllD in rr]
            rrFrac = [k/popsize for k in rr]
            plot!(subplots[4], rrFrac, results, label="A = "*string(nAAs), linecolor=RGB(0.5, 0.5, 0.5),linewidth=2, linealpha=(1-(nAAs*6.5)/popsize))
        end

        totalplot = plot(subplots[1], subplots[2], subplots[3], subplots[4], layout=l,legendfontsize=10, size=(1800, 400), dpi=1000,left_margin=11mm,bottom_margin=11mm, xlims=(0, 1.02), ylims=(minRes - 0.02, maxRes + 0.02), tickfont=font(10),xguidefontsize=20, yguidefontsize=20, titlefont=20)

        # Save the plot inside the folder
        plot_path = joinpath(folder_path, "grad_sel_alld_disc.png")
        savefig(totalplot, plot_path)

        Memoization.empty_all_caches!()
    end
end

@time begin
    plot_grad_study() 
end
