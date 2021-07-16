#--------------------------------------
# This script sets out to produce a
# basic animation of my Go8 example
#--------------------------------------

#--------------------------------------
# Author: Trent Henderson, 16 July 2021
#--------------------------------------

using Random, Plots, Distributions, StatsPlots

# Set the true probability of a student being from a Go8

p_true = 0.3

# Iterate from having seen 0 observations to 150 observations

Ns = 0:150

# Draw data from a Bernoulli distribution, i.e. draw Go8 or not

Random.seed!(12)

data = rand(Bernoulli(0.3), last(Ns))

# Our prior belief about the probability

prior_belief = Beta(3.5, 6.5)

# Make an animation.

animation = @animate for (i, N) in enumerate(Ns)

    # Count the number of Go8 vs not Go8

    heads = sum(data[1:i-1])
    tails = N - heads
    
    # Update our prior belief in closed form (this is possible because we use a conjugate prior).

    updated_belief = Beta(prior_belief.α + heads, prior_belief.β + tails)

    # Plotting
    plot(updated_belief, 
        size = (850, 450), 
        title = "Updated belief after $N observations",
        xlabel = "Possible values of actual proportion", 
        ylabel = "Probability Density", 
        legend = nothing,
        xlim = (0,1),
        fill = 0, α = 0.3, w = 3)
    vline!([p_true])
end

gif(animation, "go8anim.gif", fps = 15)
