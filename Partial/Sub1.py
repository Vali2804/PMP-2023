#sub 1 - 1
import random

def throw_coin(rigged):
    if rigged:
        return random.choice([0, 1, 1])  # 1/3 probabilitate pentru stema
    else:
        return random.choice([0, 1])  # Moneda normala

def simulare_joc():
    primul_jucator = random.choice([0, 1])  # Decide cine incepe
    if(primul_jucator==0):
        steme_p0=throw_coin(True)
        for i in range(steme_p0+1):
            steme_p1=throw_coin(False)
    else:
        steme_p1=throw_coin(False)
        for i in range(steme_p1+1):
            steme_p0=throw_coin(True)

    if steme_p0 > steme_p1:
        return 0
    else:
        return 1 

# Simulare joc de 20000 de ori
numar_jocuri = 20000
castiguri_p0 = 0
castiguri_p1 = 0

for _ in range(numar_jocuri):
    castigator = simulare_joc()
    if castigator == 0:
        castiguri_p0 += 1
    else:
        castiguri_p1 += 1

procentaj_p0 = (castiguri_p0 / numar_jocuri) * 100
procentaj_p1 = (castiguri_p1 / numar_jocuri) * 100

if procentaj_p0 > procentaj_p1:
    print("Jucatorul P0 are sansele cele mai mari de castig.")
else:
    print("Jucatorul P1 are sansele cele mai mari de castig.")

print(f"Jucatorul P0 a castigat {castiguri_p0} jocuri ({procentaj_p0}%).")
print(f"Jucatorul P1 a castigat {castiguri_p1} jocuri ({procentaj_p1}%).")


#sub 1 - 2
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
# definesc modelul folosind pgmpy
model = BayesianNetwork([('Rigged', 'Coin'), ('Coin', 'Outcome'), ('Outcome', 'Winner')])

# definesc probabilitatile conditionate
cpd_rigged = TabularCPD(variable='Rigged', variable_card=2, values=[[0.5], [0.5]])
cpd_coin = TabularCPD(variable='Coin', variable_card=2, values=[[0.5, 0.5], [0.5, 0.5]], evidence=['Rigged'], evidence_card=[2])
cpd_outcome = TabularCPD(variable='Outcome', variable_card=2, values=[[1/3, 2/3], [1/2, 1/2]], evidence=['Coin'], evidence_card=[2])
cpd_winner = TabularCPD(variable='Winner', variable_card=2, values=[[0, 1], [1, 0]], evidence=['Outcome'], evidence_card=[2])

# adaug probabilitatile conditionate la model
model.add_cpds(cpd_rigged, cpd_coin, cpd_outcome, cpd_winner)

# verific daca modelul este corect
model.check_model()

