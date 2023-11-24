# sub 2 - 1
import numpy as np
import matplotlib.pyplot as plt

medie = 10  # parametrul medie 
deviatie_standard = 2  # parametrul de deviatie standard 

# Generam 200 de timpi medii de asteptare folosind distributia normala
timpi_medii_asteptare = np.random.normal(medie, deviatie_standard, 200)

# Vizualizam distribuia generata
plt.hist(timpi_medii_asteptare, bins=20, density=True, alpha=0.6, color='g')
plt.show()

# sub 2 - 2
import pymc3 as pm

# Definirea modelului
model = pm.Model()

with model:
    # Distributii a priori pentru parametrii u și o
    prior_u = pm.Normal('prior_u', mu=0, sigma=10)
    prior_o = pm.HalfNormal('prior_o', sigma=10)

    # Distribuția a priori pentru timpii de așteptare
    prior_timpi_asteptare = pm.Normal('prior_timpi_asteptare', mu=prior_u, sigma=prior_o, shape=len(timpi_medii_asteptare))

    # Observații
    likelihood = pm.Normal('likelihood', mu=prior_timpi_asteptare, sigma=1, observed=timpi_medii_asteptare)


#Am folosit o dist normala pentru parametrul mediu u, deoarece timpul mediu de assteptare este modelat de o dist. normal,
#Am folosit o dist. HalfNormal pentru parametrul de deviatie standard o, deoarece deviatia standard trebuie să fie pozitiva, iar o dist. HalfNormal are valori pozitive.
#Distributia normala pentru prior_timpi_asteptare se extinde pentru a acoperi toti timpii de asteptare posibili.

# sub 2 - 3
import matplotlib.pyplot as plt

# nr de esantioane din dist. a posteriori
num_samples = 1000

# Sample din dist. a posteriori folosind MCMC
with model:
    trace = pm.sample(num_samples, tune=500, random_seed=42)

# Vizualizare dist. a posteriori pentru parametrul u
pm.plot_posterior(trace, var_names=['prior_u'], figsize=(10, 6))
plt.title("Distribuția a posteriori pentru u")
pm.plot_posterior(trace, var_names=['prior_u'], figsize=(10, 6))
plt.title("Distribuția a posteriori pentru u")
plt.show()
