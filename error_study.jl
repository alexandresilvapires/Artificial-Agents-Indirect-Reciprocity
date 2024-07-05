using Plots
using LinearAlgebra
using Memoization
using Dates
using SparseArrays
using ArnoldiMethod

using SpecialFunctions

using Base.Threads

# Code used to generate cooperation results for a single type of AA (eg: Disc) under a single social norm, but varying one of the 3 errors.
# We run the code for no AAs, some AAs with dynamic rep, and some G AAs.
# It is then necessary to run error_plots to generate the plots 

include("./utils.jl")
include("./strategy_dynamics.jl")

const foldername = "example_error_study"

const popsize::Int = 100  # Size of the population
const pop_sample::Int = 25 # Size of population for reputation dynamics

const fixedAARep::String = "G"              # The fixed reputations the AAs have
const canImitateAAs = true                  # If true, adaptive agents can imitate AAs

const errorAssign::Float32 = 0.01    # Prob. of assigning the wrong reputation after an observation
const errorExecut::Float32 = 0.01    # Prob. of executing the wrong action
const errorAssess::Float32 = 0.01    # Prob. of incorrectly accessing the other's reputation before choosing an action

const strengthOfSelection::Float32 = 1.0
const mutationChance::Float32 = 1.0 / popsize

# Fractions of AAs to be tested
const fracAAs::Float32 = Float32(2.0/25.0)

# Donation game parameters
const b::Float32 = 2  # benefit
const c::Int = 1  # cost of cooperation

const fraction_errors::Vector{Float32}  = utils.generate_log_spaced_values(0.0,0.5,40)

# Strategy of AAs to test. Pick between AllC, AllD or Disc
const AAsPop = "Disc"

# Social norms to test
const norm_IS_noerror::Vector = [1, 0, 1, 0]
const norm_SS_noerror::Vector = [1, 0, 1, 1]
const norm_SH_noerror::Vector = [1, 0, 0, 0]
const norm_SJ_noerror::Vector = [1, 0, 0, 1]
const all_norms = [norm_IS_noerror,norm_SS_noerror, norm_SH_noerror, norm_SJ_noerror]
const norm_labels = ["IS","SS","SH","SJ"]

norm_to_study = norm_IS_noerror

Memoization.empty_all_caches!()

global fixedAAreputations::Bool = false

function write_parameters(folder_path::String)
    parameters = (
        "Error Study" => " ",
        "Population Size" => popsize,
        "strengthOfSelection" => strengthOfSelection,
        "mutationChance" => mutationChance,
        "b" => b,
        "Error Fractions" => fraction_errors,
        "Approximation Method" => "Average Rep State with Downsampling",
        "Population Sample Size" => pop_sample
    )
    utils.write_parameters_txt(joinpath(folder_path, "parameters.txt"),parameters)
end

function get_coop_for_param(fraction_errors, norm, nAAs, index_AAs)::Vector
    result_for_error = zeros(Float32, length(fraction_errors))

    Threads.@threads for i in 1:length(fraction_errors)
        result_for_error[i] = coop_index(norm, nAAs, index_AA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, strengthOfSelection)
    end

    return result_for_error
end

function calculateErrors(norm_label::String, labelAAType::String, nAAs::Int, index_AA::Int,folder_path::String)
    for errorType in 0:2
        # First, calculate exec error
        result_for_error = zeros(Float32, length(fraction_errors))
        if errorType == 0
            result_for_error = zeros(Float32, length(fraction_errors))

            for k in eachindex(fraction_errors)
                print("\nrunning error calc SN=",norm_label, " AAs=",labelAAType," errorType=","ExecError", " errorVal=", fraction_errors[k])

                result_for_error[k] = coop_index(norm_to_study, nAAs, index_AA, errorAssess, fraction_errors[k], fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, strengthOfSelection)
                Memoization.empty_all_caches!()
            end
            utils.write_result_txt(joinpath(folder_path, "results.txt"),AAsPop*":"*norm_label*"_"*labelAAType*"ExecError",result_for_error)
        
        # Then, calculate assign error
        elseif errorType == 1
            result_for_error = zeros(Float32, length(fraction_errors))

            for k in eachindex(fraction_errors)
                print("\nrunning error calc SN=",norm_label, " AAs=",labelAAType," errorType=","AssignError", " errorVal=", fraction_errors[k])
                normWithError = utils.make_social_norm_error(norm_to_study,fraction_errors[k])

                result_for_error[k] = coop_index(normWithError, nAAs, index_AA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, strengthOfSelection)
                Memoization.empty_all_caches!()
            end
            utils.write_result_txt(joinpath(folder_path, "results.txt"),AAsPop*":"*norm_label*"_"*labelAAType*"AssignError",result_for_error)

        # Finally, calculate assess error
        else
            result_for_error = zeros(Float32, length(fraction_errors))

            for k in eachindex(fraction_errors)
                print("\nrunning error calc SN=",norm_label, " AAs=",labelAAType," errorType=","AssessError", " errorVal=", fraction_errors[k])

                result_for_error[k] = coop_index(norm_to_study, nAAs, index_AA, fraction_errors[k], errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, strengthOfSelection)
                Memoization.empty_all_caches!()
            end
            utils.write_result_txt(joinpath(folder_path, "results.txt"),AAsPop*":"*norm_label*"_"*labelAAType*"AssessError",result_for_error)
        
        end
    end
end

function plot_error_study() 
    folder_path = utils.make_plot_folder(foldername)

    write_parameters(folder_path)

    index_AA = AAsPop == "AllC" ? 0 : (AAsPop == "AllD" ? 1 : 2)
    
    labelAAType = ""

    for n in eachindex(all_norms)
        global norm_to_study = all_norms[n]
        norm_label = norm_labels[n]
        for i in 0:2
            nAAs = 0
            fixedAAreputations = false

            # First, calculate for no AAs
            if i == 0
                labelAAType = "NoAAs"
            # Then, calculate for x% dynamic-rep AAs
            elseif i == 1
                nAAs = Int.(round.(popsize .* fracAAs))
                labelAAType = "AAs"
            # Finally, calculate for x% FRAAs
            else
                nAAs = Int.(round.(popsize .* fracAAs))
                fixedAAreputations = true
                labelAAType = "FRAAs"
            end
            Memoization.empty_all_caches!()
            calculateErrors(norm_label, labelAAType, nAAs, index_AA,folder_path)
        end
        Memoization.empty_all_caches!()
    end
end

@time begin
    plot_error_study() 
end
