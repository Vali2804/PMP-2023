import numpy as np
import scipy.stats as stats

nr_sampels = 10000
lambda_poisson = 20
nr_clienti = stats.poisson.rvs(mu=lambda_poisson, size=nr_sampels)

mu_normal = 2  # media
sigma_normal = 0.5  # deviația standard
timp_plasare_plata = stats.norm.rvs(loc=mu_normal, scale=sigma_normal, size=nr_sampels) 

alpha_exponential = 25
timp_gatire = stats.expon.rvs(scale=alpha_exponential, size=nr_sampels)

#print rezultate de mai sus 
print("Rezultate simulare:")
print("-------------------")
print(f"Numărul mediu de clienți intră în restaurant într-o oră: {np.mean(nr_clienti):.2f}")
print(f"Deviația standard a numărului de clienți: {np.std(nr_clienti):.2f}")
print()
print(f"Timpul mediu de plasare și plată a unei comenzi: {np.mean(timp_plasare_plata):.2f} minute")
print(f"Deviația standard a timpului de plasare și plată: {np.std(timp_plasare_plata):.2f} minute")
print()
print(f"Timpul mediu de gătire a unei comenzi: {np.mean(timp_gatire):.2f} minute")
print(f"Deviația standard a timpului de gătire: {np.std(timp_gatire):.2f} minute")


def timp_servire_sub_15_minute(alpha, nr_sampels, lambda_poisson, mu_normal, sigma_normal):
    nr_clienti = stats.poisson.rvs(mu=lambda_poisson, size=nr_sampels)
    timp_plasare_plata = stats.norm.rvs(loc=mu_normal, scale=sigma_normal, size=nr_sampels)
    timp_gatire = stats.expon.rvs(scale=alpha, size=nr_sampels)
    timp_total = timp_plasare_plata + timp_gatire
    timp_servire_95 = np.percentile(timp_total, 95)
    return timp_servire_95

alpha_max = 0
alpha_values = np.linspace(10, 0, 1000)   
for alpha in alpha_values:
    timp_servire_95 = timp_servire_sub_15_minute(alpha, nr_sampels, lambda_poisson, mu_normal, sigma_normal)
    if timp_servire_95 <= 15:
        alpha_max = alpha
        break

if alpha_max > 0:
    print("Valoarea maximă a lui α pentru care timpul total de servire este sub 15 minute pentru 95% dintre clienți este:", alpha_max)
else:
    print("Nu s-a găsit o valoare a lui α care să îndeplinească condiția.")


timp_plasare_plata = stats.norm.rvs(loc=mu_normal, scale=sigma_normal, size=nr_sampels)
timp_gatire = stats.expon.rvs(scale=alpha_max, size=nr_sampels)

timp_asteptare_mediu = np.mean(timp_plasare_plata + timp_gatire)
print(f"Timpul mediu de așteptare pentru a fi servit este: {timp_asteptare_mediu:.2f} minute")