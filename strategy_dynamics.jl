using LinearAlgebra
using Memoization
using SparseArrays
using ArnoldiMethod
using Base.Threads

include("./utils.jl")
include("./reputation_dynamics.jl")

function all_c_receives(n_allC::Int, g_allC::Float32, n_allD::Int, g_allD::Float32, g_disc::Float32, errorAssess::Float32, errorExecut::Float32, popsize::Int)::Float32
    return (g_allC / n_allC) * (
        ((n_allC - 1) / (popsize - 1)) * coop_SG(utils.allC, errorAssess, errorExecut) +
        (n_allD / (popsize - 1)) * coop_SG(utils.allD, errorAssess, errorExecut) +
        ((popsize - n_allC - n_allD) / (popsize - 1)) * coop_SG(utils.disc, errorAssess, errorExecut)
    ) + ((n_allC - g_allC) / n_allC) * (
        ((n_allC - 1) / (popsize - 1)) * coop_SB(utils.allC, errorAssess, errorExecut) +
        (n_allD / (popsize - 1)) * coop_SB(utils.allD, errorAssess, errorExecut) +
        ((popsize - n_allC - n_allD) / (popsize - 1)) * coop_SB(utils.disc, errorAssess, errorExecut)
    )
end

function all_d_receives(n_allC::Int, g_allC::Float32, n_allD::Int, g_allD::Float32, g_disc::Float32, errorAssess::Float32, errorExecut::Float32, popsize::Int)::Float32
    return (g_allD / n_allD) * (
        (n_allC / (popsize - 1)) * coop_SG(utils.allC, errorAssess, errorExecut) +
        ((n_allD - 1) / (popsize - 1)) * coop_SG(utils.allD, errorAssess, errorExecut) +
        ((popsize - n_allC - n_allD) / (popsize - 1)) * coop_SG(utils.disc, errorAssess, errorExecut)
    ) + ((n_allD - g_allD) / n_allD) * (
        (n_allC / (popsize - 1)) * coop_SB(utils.allC, errorAssess, errorExecut) +
        ((n_allD - 1) / (popsize - 1)) * coop_SB(utils.allD, errorAssess, errorExecut) +
        ((popsize - n_allC - n_allD) / (popsize - 1)) * coop_SB(utils.disc, errorAssess, errorExecut)
    )
end

function disc_receives(n_allC::Int, g_allC::Float32, n_allD::Int, g_allD::Float32, g_disc::Float32, errorAssess::Float32, errorExecut::Float32, popsize::Int)::Float32
    return (g_disc / (popsize - n_allC - n_allD)) * (
        (n_allC / (popsize - 1)) * coop_SG(utils.allC, errorAssess, errorExecut) +
        (n_allD / (popsize - 1)) * coop_SG(utils.allD, errorAssess, errorExecut) +
        ((popsize - n_allC - n_allD - 1) / (popsize - 1)) * coop_SG(utils.disc, errorAssess, errorExecut)
    ) + ((popsize - n_allC - n_allD - g_disc) / (popsize - n_allC - n_allD)) * (
        (n_allC / (popsize - 1)) * coop_SB(utils.allC, errorAssess, errorExecut) +
        (n_allD / (popsize - 1)) * coop_SB(utils.allD, errorAssess, errorExecut) +
        ((popsize - n_allC - n_allD - 1) / (popsize - 1)) * coop_SB(utils.disc, errorAssess, errorExecut)
    )
end

function all_c_donates(n_allC::Int, g_allC::Float32, n_allD::Int, g_allD::Float32, g_disc::Float32, errorAssess::Float32, errorExecut::Float32, popsize::Int)::Float32
    return (g_allC / n_allC) * (
        (g_allC + g_allD + g_disc - 1) / (popsize - 1) * coop_SG(utils.allC, errorAssess, errorExecut) +
        (popsize - g_allC - g_allD - g_disc) / (popsize - 1) * coop_SB(utils.allC, errorAssess, errorExecut)
    ) + ((n_allC - g_allC) / n_allC) * (
        (g_allC + g_allD + g_disc) / (popsize - 1) * coop_SG(utils.allC, errorAssess, errorExecut) +
        (popsize - g_allC - g_allD - g_disc - 1) / (popsize - 1) * coop_SB(utils.allC, errorAssess, errorExecut)
    )
end

function all_d_donates(n_allC::Int, g_allC::Float32, n_allD::Int, g_allD::Float32, g_disc::Float32, errorAssess::Float32, errorExecut::Float32, popsize::Int)::Float32
    return (g_allD / n_allD) * (
        (g_allC + g_allD + g_disc - 1) / (popsize - 1) * coop_SG(utils.allD, errorAssess, errorExecut) +
        (popsize - g_allC - g_allD - g_disc) / (popsize - 1) * coop_SB(utils.allD, errorAssess, errorExecut)
    ) + ((n_allD - g_allD) / n_allD) * (
        (g_allC + g_allD + g_disc) / (popsize - 1) * coop_SG(utils.allD, errorAssess, errorExecut) +
        (popsize - g_allC - g_allD - g_disc - 1) / (popsize - 1) * coop_SB(utils.allD, errorAssess, errorExecut)
    )
end

function disc_donates(n_allC::Int, g_allC::Float32, n_allD::Int, g_allD::Float32, g_disc::Float32, errorAssess::Float32, errorExecut::Float32, popsize::Int)::Float32
    return (g_disc / (popsize - n_allC - n_allD)) * (
        (g_allC + g_allD + g_disc - 1) / (popsize - 1) * coop_SG(utils.disc, errorAssess, errorExecut) +
        (popsize - g_allC - g_allD - g_disc) / (popsize - 1) * coop_SB(utils.disc, errorAssess, errorExecut)
    ) + ((popsize - n_allC - n_allD - g_disc) / (popsize - n_allC - n_allD)) * (
        (g_allC + g_allD + g_disc) / (popsize - 1) * coop_SG(utils.disc, errorAssess, errorExecut) +
        (popsize - g_allC - g_allD - g_disc - 1) / (popsize - 1) * coop_SB(utils.disc, errorAssess, errorExecut)
    )
end

# Alex you're here fixing parameters so you can have strategy_dynamics.jl clean :3 
function fit_all_c(n_allC::Int, g_allC::Float32, n_allD::Int, g_allD::Float32, g_disc::Float32, errorAssess::Float32, errorExecut::Float32, popsize::Int, b::Float32)::Float32
    if n_allC == 0 return 0 end
    return b * all_c_receives(n_allC, g_allC, n_allD, g_allD, g_disc, errorAssess, errorExecut, popsize) - 
           1.0 * all_c_donates(n_allC, g_allC, n_allD, g_allD, g_disc, errorAssess, errorExecut, popsize)
end

function fit_all_d(n_allC::Int, g_allC::Float32, n_allD::Int, g_allD::Float32, g_disc::Float32, errorAssess::Float32, errorExecut::Float32, popsize::Int, b::Float32)::Float32
    if n_allD == 0 return 0 end
    return b * all_d_receives(n_allC, g_allC, n_allD, g_allD, g_disc, errorAssess, errorExecut, popsize) - 
           1.0 * all_d_donates(n_allC, g_allC, n_allD, g_allD, g_disc, errorAssess, errorExecut, popsize)
end

function fit_disc(n_allC::Int, g_allC::Float32, n_allD::Int, g_allD::Float32, g_disc::Float32, errorAssess::Float32, errorExecut::Float32, popsize::Int, b::Float32)::Float32
    if popsize - n_allC - n_allD == 0 return 0 end
    return b * disc_receives(n_allC, g_allC, n_allD, g_allD, g_disc, errorAssess, errorExecut, popsize) - 
           1.0 * disc_donates(n_allC, g_allC, n_allD, g_allD, g_disc, errorAssess, errorExecut, popsize)
end

@memoize function avg_fitness_all_c(n_allC::Int, n_allD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, b::Float32)::Float32
    state = get_average_rep_state(n_allC,n_allD,socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample)
    result = fit_all_c(n_allC, state[1], n_allD, state[2], state[3], errorAssess, errorExecut, popsize, b)
    return result
end

@memoize function avg_fitness_all_d(n_allC::Int, n_allD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, b::Float32)::Float32
    state = get_average_rep_state(n_allC,n_allD,socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample)
    result = fit_all_d(n_allC, state[1], n_allD, state[2], state[3], errorAssess, errorExecut, popsize, b)
    return result
end

@memoize function avg_fitness_disc(n_allC::Int, n_allD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, b::Float32)::Float32
    state = get_average_rep_state(n_allC,n_allD,socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample)
    result = fit_disc(n_allC, state[1], n_allD, state[2], state[3], errorAssess, errorExecut, popsize, b)
    return result
end

function p_imit(fA::Float32, fB::Float32, strengthOfSelection::Float32)::Float32
    return (1 + exp(-strengthOfSelection * (fB - fA))) ^ -1
end

function damp_factor_all_c(n_allC::Int, n_allD::Int, nAAs::Int, indexOfAA::Int, popsize::Int)::Float32
    if indexOfAA == 0
        if n_allC < nAAs
            return 0
        else
            if n_allC == 0
                return 1
            else
                return (n_allC-nAAs)/n_allC
            end
        end
    else
        return 1
    end
end

function damp_factor_all_d(n_allC::Int, n_allD::Int, nAAs::Int, indexOfAA::Int, popsize::Int)::Float32
    if indexOfAA == 1
        if n_allD < nAAs
            return 0
        else
            if n_allD == 0
                return 1
            else
                return (n_allD-nAAs)/n_allD
            end
        end
    else
        return 1
    end
end

function damp_factor_disc(n_allC::Int, n_allD::Int, nAAs::Int, indexOfAA::Int, popsize::Int)::Float32
    if indexOfAA == 2
        if popsize - n_allC - n_allD < nAAs
            return 0
        else
            if popsize - n_allC - n_allD == 0
                return 1
            else
                return (popsize - n_allC - n_allD-nAAs)/(popsize - n_allC - n_allD)
            end
        end
    else
        return 1
    end
end

function n_rolemodels(n_allC::Int, n_allD::Int, nAAs::Int, indexOfAA::Int, stratToImit::Int, popsize::Int, canImitateAAs::Bool)::Float32
    if (!canImitateAAs && stratToImit == indexOfAA)
        return stratToImit == 0 ? ((n_allC - nAAs) / (popsize - 1 - nAAs)) : (stratToImit == 1 ? ((n_allD - nAAs) / (popsize - 1 - nAAs)) : ((popsize - n_allC - n_allD - nAAs) / (popsize - 1 - nAAs)))
    else
        return stratToImit == 0 ? (n_allC / (popsize - nAAs - 1)) : (stratToImit == 1 ? (n_allD / (popsize - nAAs - 1)) : ((popsize - n_allC - n_allD) / (popsize  - nAAs - 1)))
    end
end

function t_all_d_all_c(n_allC::Int, n_allD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, canImitateAAs::Bool, b::Float32, mutationChance::Float32, SoS::Float32)::Float32
    return damp_factor_all_d(n_allC, n_allD, nAAs, indexOfAA, popsize) * (
            (1 - mutationChance) * (n_allD / popsize) * n_rolemodels(n_allC, n_allD, nAAs, indexOfAA, 0, popsize, canImitateAAs) *
            p_imit(avg_fitness_all_d(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, b),
                avg_fitness_all_c(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, b), SoS) +
            mutationChance * n_allD / (2 * popsize)
    )
end

function t_disc_all_c(n_allC::Int, n_allD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, canImitateAAs::Bool, b::Float32, mutationChance::Float32,  SoS::Float32)::Float32
    return damp_factor_disc(n_allC, n_allD, nAAs, indexOfAA, popsize) * (
            (1 - mutationChance) * ((popsize - n_allD - n_allC) / popsize) * n_rolemodels(n_allC, n_allD, nAAs, indexOfAA, 0, popsize, canImitateAAs) *
            p_imit(avg_fitness_disc(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, b),
                avg_fitness_all_c(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, b), SoS) +
            mutationChance * (popsize - n_allD - n_allC) / (2 * popsize)
    )
end

function t_all_c_all_d(n_allC::Int, n_allD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, canImitateAAs::Bool, b::Float32, mutationChance::Float32,  SoS::Float32)::Float32
    return damp_factor_all_c(n_allC, n_allD, nAAs, indexOfAA, popsize) * (
            (1 - mutationChance) * (n_allC / popsize) * n_rolemodels(n_allC, n_allD, nAAs, indexOfAA, 1, popsize, canImitateAAs) *
            p_imit(avg_fitness_all_c(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, b),
                avg_fitness_all_d(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, b), SoS) +
            mutationChance * n_allC / (2 * popsize)
    )
end


function t_disc_all_d(n_allC::Int, n_allD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, canImitateAAs::Bool, b::Float32, mutationChance::Float32,  SoS::Float32)::Float32
    return damp_factor_disc(n_allC, n_allD, nAAs, indexOfAA, popsize) * (
            (1 - mutationChance) * ((popsize - n_allD - n_allC) / popsize) * n_rolemodels(n_allC, n_allD, nAAs, indexOfAA, 1, popsize, canImitateAAs) *
            p_imit(avg_fitness_disc(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, b),
                avg_fitness_all_d(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, b), SoS) +
            mutationChance * (popsize - n_allD - n_allC) / (2 * popsize)
    )
end

function t_all_c_disc(n_allC::Int, n_allD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, canImitateAAs::Bool, b::Float32, mutationChance::Float32,  SoS::Float32)::Float32
    return damp_factor_all_c(n_allC, n_allD, nAAs, indexOfAA, popsize) * (
            (1 - mutationChance) * (n_allC / popsize) * n_rolemodels(n_allC, n_allD, nAAs, indexOfAA, 2, popsize, canImitateAAs) *
            p_imit(avg_fitness_all_c(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, b),
                avg_fitness_disc(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, b), SoS) +
            mutationChance * n_allC / (2 * popsize)
    )
end

function t_all_d_disc(n_allC::Int, n_allD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, canImitateAAs::Bool, b::Float32, mutationChance::Float32,  SoS::Float32)::Float32
    return damp_factor_all_d(n_allC, n_allD, nAAs, indexOfAA, popsize) * (
            (1 - mutationChance) * (n_allD / popsize) * n_rolemodels(n_allC, n_allD, nAAs, indexOfAA, 2, popsize, canImitateAAs) *
            p_imit(avg_fitness_all_d(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, b),
                avg_fitness_disc(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, b), SoS) +
            mutationChance * n_allD / (2 * popsize)
    )
end

function grad_disc_alld(n_allC::Int, n_allD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, canImitateAAs::Bool, b::Float32, mutationChance::Float32,  SoS::Float32)::Float32
    return t_all_d_disc(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, SoS) - t_disc_all_d(n_allC, n_allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, SoS)
end

@memoize function get_states_strat(nAAs::Int,indexOfAA::Int, popsize::Int)::Vector{Tuple{Int, Int, Int}}
    states = [(nAllC, nAllD, popsize - nAllC - nAllD) 
    for nAllC in 0:popsize 
    for nAllD in 0:(popsize - nAllC) 
    if nAllC + nAllD <= popsize]

    states = [state for state in states 
        if (state[indexOfAA+1] >= nAAs)]

    return states
end

@memoize function stationary_dist_strategy(socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, canImitateAAs::Bool, b::Float32, mutationChance::Float32,  SoS::Float32)::Vector{Float32}
    states::Vector{Tuple{Int, Int, Int}} = get_states_strat(nAAs, indexOfAA, popsize)
    lookup = utils.create_lookup_table(states)
    
    transition_matrix = spzeros(length(states),length(states))

    currentPos = 0
    t1, t2, t3, t4, t5, t6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in 0:popsize
        for j in 0:(popsize - i)
            k = popsize - j - i
            t1, t2, t3, t4, t5, t6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            if k < 0
                continue
            end

            if indexOfAA == 0 && i < nAAs
                continue
            end
            if indexOfAA == 1 && j < nAAs
                continue
            end
            if indexOfAA == 2 && k < nAAs
                continue
            end

            currentPos = utils.pos_of_state(lookup, (i, j, k))

            if i < popsize && ((indexOfAA == 1 && j > nAAs) || (indexOfAA != 1 && j > 0))
                transition_matrix[currentPos,utils.pos_of_state(lookup, (i+1, j-1, k))] = t1 = t_all_d_all_c(i, j, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, SoS)
            end

            if i < popsize && ((indexOfAA == 2 && k > nAAs) || (indexOfAA != 2 && k > 0))
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i+1, j, k-1))] = t2 = t_disc_all_c(i, j, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, SoS)
            end

            if ((indexOfAA == 0 && i > nAAs) || (indexOfAA != 0 && i > 0)) && j < popsize
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i-1, j+1, k))] = t3 = t_all_c_all_d(i, j, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, SoS)
            end

            if ((indexOfAA == 0 && i > nAAs) || (indexOfAA != 0 && i > 0)) && k < popsize
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i-1, j, k+1))] = t4 = t_all_c_disc(i, j, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, SoS)
            end

            if ((indexOfAA == 2 && k > nAAs) || (indexOfAA != 2 && k > 0)) && j < popsize
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i, j+1, k-1))] = t5 = t_disc_all_d(i, j, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, SoS)
            end

            if ((indexOfAA == 1 && j > nAAs) || (indexOfAA != 1 && j > 0)) && k < popsize
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i, j-1, k+1))] = t6 = t_all_d_disc(i, j, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, SoS)
            end

            transition_matrix[currentPos,currentPos] = 1 - t1 - t2 - t3 - t4 - t5 - t6
        end
    end

    result = utils.get_transition_matrix_statdist(transition_matrix)

    return result
end

function stat_dist_at_point_strat(n_allC::Int, n_allD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, canImitateAAs::Bool, b::Float32, mutationChance::Float32,  SoS::Float32)::Float32
    states = get_states_strat(nAAs, indexOfAA, popsize)

    if (n_allC, n_allD, popsize-n_allC-n_allD) ∉ states
        return 0
    end

    stat_dist = stationary_dist_strategy(socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, SoS)
    pos = utils.find_pos_of_state(states, (n_allC, n_allD, popsize-n_allC-n_allD))
    result = stat_dist[pos]

    return result
end


# Generate all possible states
function make_all_states(popsize::Int)::Vector{Tuple{Int, Int, Int}}
    return [(nAllC, nAllD, popsize - nAllC - nAllD) for nAllC in 0:popsize, nAllD in 0:popsize if nAllC + nAllD <= popsize]
end 

function all_AA_states_valid(nAAs::Int, index_of_AA::Int, popsize::Int)::Vector{Tuple{Int, Int, Int}}
    return [state for state in make_all_states(popsize) if state[index_of_AA+1] >= nAAs]
end

function avg_coop_allC(nAllC::Int, nAllD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int)::Float32
    if nAllC == 0
        return 0
    end
    state = get_average_rep_state(nAllC,nAllD,socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample)
    result = all_c_donates(nAllC, state[1], nAllD, state[2], state[3], errorAssess, errorExecut, popsize)
    return result
end

function avg_coop_allD(nAllC::Int, nAllD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int)::Float32
    if nAllD == 0
        return 0
    end
    state = get_average_rep_state(nAllC,nAllD,socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample)
    result = all_d_donates(nAllC, state[1], nAllD, state[2], state[3], errorAssess, errorExecut, popsize)
    return result
end

function avg_coop_disc(nAllC::Int, nAllD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int)::Float32
    if popsize - nAllC - nAllD == 0
        return 0
    end
    state = get_average_rep_state(nAllC,nAllD,socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample)
    result = disc_donates(nAllC, state[1], nAllD, state[2], state[3], errorAssess, errorExecut, popsize)
    return result
end

function sum_coop(allC::Int, allD::Int, socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, canImitateAAs::Bool, b::Float32, mutationChance::Float32,  SoS::Float32)::Float32
    state_index = findfirst(x -> x == (allC, allD, popsize - allC - allD), all_AA_states_valid(nAAs, indexOfAA, popsize))

    if state_index !== nothing
        stationaryDist = stat_dist_at_point_strat(allC, allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, SoS)
        res =  (stationaryDist * (
            avg_coop_allC(allC, allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample) * allC +
            avg_coop_allD(allC, allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample) * allD +
            avg_coop_disc(allC, allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample) * (popsize - allC - allD)
        )) / popsize
        return res
    else
        return 0.0
    end
end

function avg_reputation(socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, canImitateAAs::Bool, b::Float32, mutationChance::Float32,  SoS::Float32)::Vector
    states = all_AA_states_valid(nAAs, indexOfAA, popsize)
    avg_rep = [Float32(0),Float32(0),Float32(0)]

    for state in states
        stationaryDist = stat_dist_at_point_strat(state[1], state[2], socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, SoS)
        avg_rep_at_state = get_average_rep_state(state[1], state[2], socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample)

        avg_rep .+= stationaryDist .* avg_rep_at_state
    end

    return avg_rep
end

function coop_index(socialnorm::Vector, nAAs::Int, indexOfAA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int, canImitateAAs::Bool, b::Float32, mutationChance::Float32,  SoS::Float32)::Float32
    coopIndex_value = 0.0
    for allC in 0:popsize
        for allD in 0:(popsize - allC)
            if (indexOfAA == 0 && allC >= nAAs) || (indexOfAA == 1 && allD >= nAAs) || (indexOfAA == 2 && popsize - allC - allD >= nAAs)
                coopIndex_value += sum_coop(allC, allD, socialnorm, nAAs, indexOfAA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, popsize, pop_sample, canImitateAAs, b, mutationChance, SoS)
            end
        end
    end
    return coopIndex_value
end

# Example usage to get cooperation index:
# const errorGeneral::Float32 = 0.01
# const normToTry = utils.make_social_norm_error([1,0,1,0], errorGeneral)
# const bToUse::Float32 = 3.0
# const mutChance::Float32 = 0.01
# const sos::Float32 = 1.0
# println(coop_index(normToTry, 2, 2, errorGeneral, errorGeneral, true, "G", 100, 20, true, bToUse, mutChance, sos))
# Memoization.empty_all_caches!()