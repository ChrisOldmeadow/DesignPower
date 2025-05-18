"""
Stepped wedge simulation for cluster randomized trials using Julia.

This module provides high-performance simulation functions for
stepped wedge cluster randomized trial designs.
"""

using Random
using Distributions
using Statistics
using LinearAlgebra

"""
    simulate_stepped_wedge(clusters, steps, individuals_per_cluster, icc, 
                          treatment_effect, std_dev, nsim=1000, alpha=0.05)

Simulate a stepped wedge cluster randomized trial and estimate power.

# Arguments
- `clusters::Int`: Number of clusters
- `steps::Int`: Number of time steps (including baseline)
- `individuals_per_cluster::Int`: Number of individuals per cluster per time step
- `icc::Float64`: Intracluster correlation coefficient
- `treatment_effect::Float64`: Effect size of the intervention
- `std_dev::Float64`: Total standard deviation of outcome
- `nsim::Int=1000`: Number of simulations
- `alpha::Float64=0.05`: Significance level

# Returns
Dictionary containing the estimated power and simulation details
"""
function simulate_stepped_wedge(clusters::Int, steps::Int, individuals_per_cluster::Int,
                               icc::Float64, treatment_effect::Float64, std_dev::Float64,
                               nsim::Int=1000, alpha::Float64=0.05)
    
    # Calculate between-cluster and within-cluster variance
    var_between = icc * std_dev^2
    var_within = (1 - icc) * std_dev^2
    
    # Initialize counter for significant results
    sig_count = 0
    
    # Store p-values for all simulations
    p_values = Float64[]
    
    # Create distributions for sampling
    between_dist = Normal(0, sqrt(var_between))
    within_dist = Normal(0, sqrt(var_within))
    
    # Run simulations
    for _ in 1:nsim
        # Create arrays to store data
        all_cluster_ids = Int[]
        all_time_ids = Int[]
        all_treatments = Bool[]
        all_outcomes = Float64[]
        
        # Generate cluster random effects
        cluster_effects = rand(between_dist, clusters)
        
        # Assign clusters to steps (excluding baseline)
        cluster_step_assignment = shuffle([i % (steps - 1) + 1 for i in 0:(clusters-1)])
        
        # Generate data
        for c in 1:clusters
            # Determine when this cluster receives intervention
            step_of_intervention = cluster_step_assignment[c]
            
            # Get cluster effect for this cluster
            cluster_effect = cluster_effects[c]
            
            for t in 1:steps
                # Determine if cluster is under treatment at this time step
                is_treated = t >= step_of_intervention
                
                # Generate outcomes for individuals in this cluster at this time step
                for _ in 1:individuals_per_cluster
                    # Calculate mean for this individual
                    individual_mean = 0 + (is_treated ? treatment_effect : 0) + cluster_effect
                    
                    # Generate outcome
                    outcome = individual_mean + rand(within_dist)
                    
                    # Add to dataset
                    push!(all_cluster_ids, c)
                    push!(all_time_ids, t)
                    push!(all_treatments, is_treated)
                    push!(all_outcomes, outcome)
                end
            end
        end
        
        # Convert to arrays
        treatments = Int.(all_treatments)
        outcomes = Float64.(all_outcomes)
        
        # Perform simple t-test (this is a simplification; in practice a mixed model would be used)
        treated_outcomes = outcomes[treatments .== 1]
        control_outcomes = outcomes[treatments .== 0]
        
        # Calculate t-test manually
        n1 = length(treated_outcomes)
        n2 = length(control_outcomes)
        mean1 = mean(treated_outcomes)
        mean2 = mean(control_outcomes)
        var1 = var(treated_outcomes)
        var2 = var(control_outcomes)
        
        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        
        # Standard error
        se = sqrt(pooled_var * (1/n1 + 1/n2))
        
        # t-statistic
        t_stat = (mean1 - mean2) / se
        
        # Degrees of freedom
        df = n1 + n2 - 2
        
        # p-value
        p_val = 2 * (1 - cdf(TDist(df), abs(t_stat)))
        
        push!(p_values, p_val)
        
        # Check if result is significant
        if p_val < alpha
            sig_count += 1
        end
    end
    
    # Calculate power
    power = sig_count / nsim
    
    # Return results as a dictionary (for compatibility with Python)
    return Dict(
        "power" => power,
        "mean_p_value" => mean(p_values),
        "median_p_value" => median(p_values),
        "nsim" => nsim,
        "parameters" => Dict(
            "clusters" => clusters,
            "steps" => steps,
            "individuals_per_cluster" => individuals_per_cluster,
            "total_n" => clusters * steps * individuals_per_cluster,
            "icc" => icc,
            "treatment_effect" => treatment_effect,
            "std_dev" => std_dev,
            "alpha" => alpha
        )
    )
end

"""
    call_from_python_example()

Example showing how to call the stepped wedge simulation from Python.
"""
function call_from_python_example()
    println("Example of calling the stepped wedge simulation from Python:")
    println("from julia import Main")
    println("from julia.api import Julia")
    println("")
    println("# Initialize Julia")
    println("jl = Julia(compiled_modules=False)")
    println("Main.include(\"julia_backend/stepped_wedge.jl\")")
    println("")
    println("# Call the Julia function")
    println("result = Main.simulate_stepped_wedge(")
    println("    12,  # clusters")
    println("    4,   # steps")
    println("    10,  # individuals per cluster")
    println("    0.05,  # ICC")
    println("    0.5,   # treatment effect")
    println("    1.0,   # standard deviation")
    println("    1000,  # number of simulations")
    println("    0.05   # alpha")
    println(")")
    println("")
    println("print(f\"Estimated power: {result['power']:.4f}\")")
end
