import pymc3 as pm
import arviz as az
import numpy as np

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

with pm.Model() as model:
    n = pm.Poisson('n', mu=10)
    
    for Y in Y_values:
        for theta in theta_values:
            Y_observed = pm.Binomial('Y_observed', n=n, p=theta, observed=Y)
    

    trace = pm.sample(1000, tune=1000)

az.plot_posterior(trace)
