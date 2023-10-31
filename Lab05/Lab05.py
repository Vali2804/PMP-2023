import pymc3 as pm
import numpy as np
import pandas as pd

traffic_data = pd.read_csv('trafic.csv', header=0, names=['minute', 'traffic'])

intervals = [(7, 8), (8, 9), (16, 17), (19, 20), (20, 21)]

if __name__ == '__main__':
    with pm.Model() as model:

        lamb = pm.Exponential('lamb', lam=1)
        traffic = pm.Poisson('traffic', mu=lamb, observed=traffic_data['traffic'])
        trace = pm.sample(10000, tune=1000, chains=4)

mean_lambdas = []
for interval in intervals:
    interval_data = traffic_data[(traffic_data['minute'] >= interval[0]*60) & (traffic_data['minute'] < interval[1]*60)]
    mean_traffic = interval_data['traffic'].mean()
    mean_lambdas.append(mean_traffic)

for i, interval in enumerate(intervals):
    print(f'Interval {interval}: Estimated lambda = {mean_lambdas[i]:.2f}')

pm.summary(trace, var_names=['lamb'])
