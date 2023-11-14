import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import arviz as az


#subpct. a)
raw_data = pd.read_csv('auto-mpg.csv')
data=raw_data[['mpg','horsepower']]
data=data.drop(data[data.horsepower == '?'].index)
print(data)
x = data['horsepower'].values
y = data['mpg'].values

x=np.array(x,dtype=np.float64)
y=np.array(y,dtype=np.float64)

plt.scatter(x, y)
plt.xlabel('mpg')
plt.ylabel('horsepower')
plt.show()

#subpct. b)
with pm.Model() as model_poly:
    alpha = pm.Normal('alpha', mu=y.mean(), sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=1)
    eps = pm.HalfCauchy('eps', 5)
    mu = pm.Deterministic('mu', alpha + beta * x)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y)
    idata_poly = pm.sample(500, return_inferencedata=True)



#subpct. c)
plt.plot(x, y, 'C0.')
posterior_g = idata_poly.posterior.stack(samples={"chain", "draw"})
alpha_m = posterior_g['alpha'].mean().item()
beta_m = posterior_g['beta'].mean().item()
draws = range(0, posterior_g.samples.size, 10)
plt.plot(x, posterior_g['alpha'][draws].values + posterior_g['beta'][draws].values * x[:,None],c='gray', alpha=0.5)
plt.plot(x, alpha_m + beta_m * x, c='k',label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()


#subpct. d)
Ppc = pm.sample_posterior_predictive(idata_poly, model=model_poly)
plt.plot(x, y, 'b.')
plt.plot(x, alpha_m + beta_m * x, c='k',
label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
az.plot_hdi(x, Ppc.posterior_predictive['y_pred'], hdi_prob=0.95, color='gray')
az.plot_hdi(x, Ppc.posterior_predictive['y_pred'], color='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


#Se poate observa ca daca valoarea "Horsepower" creste, valoarea "mpg" scade, avand un consum mai mare.