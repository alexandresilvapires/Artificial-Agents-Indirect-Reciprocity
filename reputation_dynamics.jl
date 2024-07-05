using LinearAlgebra
using Memoization
using Dates
using SparseArrays
using ArnoldiMethod

include("./utils.jl")

# Chances of cooperation when user of strat S sees a G, considering errors
@memoize function coop_SG(strategy::Vector, errorAssess::Float32, errorExecut::Float32)::Float32
    strategy_after_error = utils.add_exec_error(strategy, errorExecut)
    return (1 - errorAssess) * strategy_after_error[1] + errorAssess * strategy_after_error[2]
end

@memoize function coop_SB(strategy::Vector, errorAssess::Float32, errorExecut::Float32)::Float32
    strategy_after_error = utils.add_exec_error(strategy, errorExecut)
    return errorAssess * strategy_after_error[1] + (1 - errorAssess) * strategy_after_error[2]
end

# Reputation update prob
function assign_GG(strategy::Vector, socialnorm::Vector, errorAssess::Float32, errorExecut::Float32)::Float32
    return (1 - errorAssess) * (coop_SG(strategy, errorAssess, errorExecut) * socialnorm[1] + (1 - coop_SG(strategy, errorAssess, errorExecut)) * socialnorm[2]) +
           errorAssess * (coop_SG(strategy, errorAssess, errorExecut) * socialnorm[3] + (1 - coop_SG(strategy, errorAssess, errorExecut)) * socialnorm[4])
end

function assign_GB(strategy::Vector, socialnorm::Vector, errorAssess::Float32, errorExecut::Float32)::Float32
    return errorAssess * (coop_SB(strategy, errorAssess, errorExecut) * socialnorm[1] + (1 - coop_SB(strategy, errorAssess, errorExecut)) * socialnorm[2]) +
           (1 - errorAssess) * (coop_SB(strategy, errorAssess, errorExecut) * socialnorm[3] + (1 - coop_SB(strategy, errorAssess, errorExecut)) * socialnorm[4])
end

# Birth death probabilities
function common_factor_T_plus(strategy::Vector, g_allC::Int, g_allD::Int, g_disc::Int, socialnorm::Vector,populationsize::Int, errorAssess::Float32, errorExecut::Float32)::Float32
    return ((g_allC + g_allD + g_disc) / (populationsize - 1)) * assign_GG(strategy, socialnorm, errorAssess, errorExecut) +
            ((populationsize - g_allC - g_allD - g_disc - 1) / (populationsize - 1)) * assign_GB(strategy, socialnorm, errorAssess, errorExecut)
end

function common_factor_T_minus(strategy::Vector, g_allC::Int, g_allD::Int, g_disc::Int, socialnorm::Vector,populationsize::Int, errorAssess::Float32, errorExecut::Float32)::Float32
    return ((g_allC + g_allD + g_disc - 1) / (populationsize - 1)) * (1 - assign_GG(strategy, socialnorm, errorAssess, errorExecut)) +
            ((populationsize - g_allC - g_allD - g_disc) / (populationsize - 1)) * (1 - assign_GB(strategy, socialnorm, errorAssess, errorExecut))
end

# Damping factors
function damp_factor_plus(pop::Int, n_AAs::Int, index_of_AA::Int, fixedAAreputations::Bool, fixedAARep::String)::Float32
    if fixedAAreputations && index_of_AA == pop
        return fixedAARep == "G" ? 0 : n_AAs
    end
    return 0
end

function damp_factor_minus(pop::Int, n_AAs::Int, index_of_AA::Int, fixedAAreputations::Bool, fixedAARep::String)::Float32
    if fixedAAreputations && index_of_AA == pop
        return fixedAARep == "B" ? 0 : n_AAs
    end
    return 0
end

function t_plus_rep_all_c(n_allC::Int, g_allC::Int, g_allD::Int, g_disc::Int, socialnorm::Vector, n_AAs::Int, index_of_AA::Int,populationsize::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String)::Float32
    return ((n_allC - g_allC - damp_factor_plus(0, n_AAs, index_of_AA, fixedAAreputations, fixedAARep)) / populationsize) *
    common_factor_T_plus(utils.allC, g_allC, g_allD, g_disc, socialnorm,populationsize, errorAssess, errorExecut)
end

function t_minus_rep_all_c(g_allC::Int, g_allD::Int, g_disc::Int, socialnorm::Vector, n_AAs::Int, index_of_AA::Int,populationsize::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String)::Float32
    return ((g_allC - damp_factor_minus(0, n_AAs, index_of_AA, fixedAAreputations, fixedAARep)) / populationsize) *
        common_factor_T_minus(utils.allC, g_allC, g_allD, g_disc, socialnorm,populationsize, errorAssess, errorExecut)
end

function t_plus_rep_all_d(g_allC::Int, n_allD::Int, g_allD::Int, g_disc::Int, socialnorm::Vector, n_AAs::Int, index_of_AA::Int,populationsize::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String)::Float32
    return ((n_allD - g_allD - damp_factor_plus(1, n_AAs, index_of_AA, fixedAAreputations, fixedAARep)) / populationsize) *
        common_factor_T_plus(utils.allD, g_allC, g_allD, g_disc, socialnorm,populationsize, errorAssess, errorExecut)
end

function t_minus_rep_all_d(g_allC::Int, g_allD::Int, g_disc::Int, socialnorm::Vector, n_AAs::Int, index_of_AA::Int,populationsize::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String)::Float32
    return ((g_allD - damp_factor_minus(1, n_AAs, index_of_AA, fixedAAreputations, fixedAARep)) / populationsize) *
        common_factor_T_minus(utils.allD, g_allC, g_allD, g_disc, socialnorm,populationsize, errorAssess, errorExecut)
end

function t_plus_rep_disc(n_allC::Int, g_allC::Int, n_allD::Int, g_allD::Int, g_disc::Int, socialnorm::Vector, n_AAs::Int, index_of_AA::Int,populationsize::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String)::Float32
    return ((populationsize - n_allC - n_allD - g_disc - damp_factor_plus(2, n_AAs, index_of_AA, fixedAAreputations, fixedAARep)) / populationsize) *
        common_factor_T_plus(utils.disc, g_allC, g_allD, g_disc, socialnorm,populationsize, errorAssess, errorExecut)
end

function t_minus_rep_disc(g_allC::Int, g_allD::Int, g_disc::Int, socialnorm::Vector, n_AAs::Int, index_of_AA::Int,populationsize::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String)::Float32
    return ((g_disc - damp_factor_minus(2, n_AAs, index_of_AA, fixedAAreputations, fixedAARep)) / populationsize) *
        common_factor_T_minus(utils.disc, g_allC, g_allD, g_disc, socialnorm,populationsize, errorAssess, errorExecut)
end

function get_pop_number(n_allC::Int, n_allD::Int, index_of_AA::Int,populationsize::Int)::Int
    if index_of_AA == 0
        return n_allC
    elseif index_of_AA == 1
        return n_allD
    end
    return populationsize - n_allC - n_allD
end

function get_states(n_allC::Int, n_allD::Int,populationsize::Int)::Vector{Tuple{Int, Int, Int}}
    return [(i, j, k) for i in 0:n_allC for j in 0:n_allD for k in 0:(populationsize - n_allC - n_allD)]
end

function get_filtered_rep_states(n_allC, n_allD, n_AAs, index_of_AA,populationsize::Int, fixedAAreputations::Bool, fixedAARep::String)::Vector{Tuple{Int, Int, Int}}
    if fixedAAreputations
        return [(i, j, k) for i in 0:n_allC for j in 0:n_allD for k in 0:(populationsize - n_allC - n_allD)
                if (fixedAARep == "G" && (i,j,k)[index_of_AA+1] >= n_AAs) || (fixedAARep == "B" && get_pop_number(n_allC, n_allD, index_of_AA,populationsize) - (i,j,k)[index_of_AA+1] >= n_AAs)]
    end
    return get_states(n_allC, n_allD,populationsize)
end

function calculate_stationary_dist_rep(n_allC::Int, n_allD::Int, socialnorm::Vector, n_AAs::Int, index_of_AA::Int,populationsize::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String)::Vector{Float32}
    states::Vector{Tuple{Int, Int, Int}} = get_filtered_rep_states(n_allC, n_allD, n_AAs, index_of_AA, populationsize, fixedAAreputations, fixedAARep)
    lookup = utils.create_lookup_table(states)
    
    transition_matrix = spzeros(length(states),length(states))

    currentPos = 0

    t1, t2, t3, t4, t5, t6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in 0:n_allC
        for j in 0:n_allD
            for k in 0:(populationsize - n_allC - n_allD)

                # Don't calculate illegal fixed rep states
                if fixedAAreputations && !((i, j, k) in states)
                    continue
                end

                t1, t2, t3, t4, t5, t6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

                currentPos = utils.pos_of_state(lookup, (i, j, k))

                # Add or remove allC
                if i < n_allC && (i + 1, j, k) in states
                    transition_matrix[currentPos, utils.pos_of_state(lookup, (i + 1, j, k))] = t1 =
                        t_plus_rep_all_c(n_allC, i, j, k, socialnorm, n_AAs, index_of_AA,populationsize, errorAssess, errorExecut, fixedAAreputations, fixedAARep)
                end
                if i > 0 && (i - 1, j, k) in states
                    transition_matrix[currentPos, utils.pos_of_state(lookup, (i - 1, j, k))] = t2 =
                        t_minus_rep_all_c(i, j, k, socialnorm, n_AAs, index_of_AA,populationsize, errorAssess, errorExecut, fixedAAreputations, fixedAARep)
                end

                # Add or remove allD
                if j < n_allD && (i, j + 1, k) in states
                    transition_matrix[currentPos, utils.pos_of_state(lookup, (i, j + 1, k))] = t3 =
                        t_plus_rep_all_d(i, n_allD, j, k, socialnorm, n_AAs, index_of_AA,populationsize, errorAssess, errorExecut, fixedAAreputations, fixedAARep)
                end
                if j > 0 && (i, j - 1, k) in states
                    transition_matrix[currentPos, utils.pos_of_state(lookup, (i, j - 1, k))] = t4 =
                        t_minus_rep_all_d(i, j, k, socialnorm, n_AAs, index_of_AA,populationsize, errorAssess, errorExecut, fixedAAreputations, fixedAARep)
                end

                # Add or remove disc
                if k < populationsize - n_allC - n_allD && (i, j, k + 1) in states
                    transition_matrix[currentPos, utils.pos_of_state(lookup, (i, j, k + 1))] = t5 =
                        t_plus_rep_disc(n_allC, i, n_allD, j, k, socialnorm, n_AAs, index_of_AA,populationsize, errorAssess, errorExecut, fixedAAreputations, fixedAARep)
                end
                if k > 0 && (i, j, k - 1) in states
                    transition_matrix[currentPos, utils.pos_of_state(lookup, (i, j, k - 1))] = t6 =
                        t_minus_rep_disc(i, j, k, socialnorm, n_AAs, index_of_AA,populationsize, errorAssess, errorExecut, fixedAAreputations, fixedAARep)
                end

                # Stay in the same state
                transition_matrix[currentPos, currentPos] =
                    1 - t1 - t2 - t3 - t4 - t5 - t6
            end
        end
    end

    result = utils.get_transition_matrix_statdist(transition_matrix)

    return result
end

@memoize function get_average_rep_state_aux(n_allC::Int, n_allD::Int, socialnorm::Vector, n_AAs::Int, index_of_AA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, pop_sample::Int)::Tuple{Float32,Float32,Float32}
    # Assumes scaled nAllC, nAllD and nAAs.
    
    states = get_filtered_rep_states(n_allC, n_allD, n_AAs, index_of_AA, pop_sample, fixedAAreputations, fixedAARep)
    stat_dist = calculate_stationary_dist_rep(n_allC, n_allD, socialnorm, n_AAs, index_of_AA,pop_sample, errorAssess, errorExecut, fixedAAreputations, fixedAARep)
    avg_state = [Float32(0), Float32(0), Float32(0)]

    for i in eachindex(states)
        avg_state .+= states[i] .* stat_dist[i]
    end

    return tuple(avg_state...)
end

function get_average_rep_state(n_allC::Int, n_allD::Int, socialnorm::Vector, n_AAs::Int, index_of_AA::Int, errorAssess::Float32, errorExecut::Float32, fixedAAreputations::Bool, fixedAARep::String, popsize::Int, pop_sample::Int)::Tuple{Float32,Float32,Float32}
    # We call this first so we can use the memoized result in case roundings lead to the same sampled population as another case
    
    population_ratio::Float32 = Float32(pop_sample)/popsize
    new_nAAs::Int = round(n_AAs * population_ratio)
    new_nAllC::Int = round(n_allC * population_ratio)
    new_nAllD::Int = round(n_allD * population_ratio)

    if pop_sample - new_nAllC - new_nAllD < 0
        if new_nAllC > new_nAllD
            new_nAllC -= 1
        else
            new_nAllD -= 1
        end
    end
    newpop::Tuple = (new_nAllC,new_nAllD, pop_sample - new_nAllC - new_nAllD)

    if newpop[index_of_AA + 1] < new_nAAs
        new_nAAs = newpop[index_of_AA + 1]
    end
    if new_nAAs < 0
        new_nAAs = 0
    end

    allC_ratio = Float32(new_nAllC)/n_allC
    allD_ratio = Float32(new_nAllD)/n_allD
    disc_ratio = Float32(pop_sample-new_nAllC-new_nAllD)/(popsize-n_allC-n_allD)

    result = collect(get_average_rep_state_aux(new_nAllC,new_nAllD, socialnorm,new_nAAs,index_of_AA, errorAssess, errorExecut, fixedAAreputations, fixedAARep, pop_sample))

    result ./= [allC_ratio, allD_ratio, disc_ratio]
    result = [isnan(x) ? 1 : x for x in result] # Change cases for scaling by 0

    return tuple(result...)
end

# Example usage to measure reputation in a certain state:
#const errorTest::Float32 = 0.01
#println(get_average_rep_state(5,10, [1,0,0,1], 2, 2, errorTest, errorTest, false, "G", 20, 10))
#Memoization.empty_all_caches!()