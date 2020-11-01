# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 06:50:33 2020

@author: Amel
@author: Esteban
@author: Marco
"""


# Código proyecto de riesgo

#%%
import pandas as pd
import numpy as np
from scipy.stats import pareto
import matplotlib.pyplot as plt
from fitter import Fitter
import seaborn as sns
import statsmodels.api as sm

datos = pd.read_excel("Defraudaciones enero-junio 2020.xlsx")



#%%

print(datos.columns)

plt.hist(datos['MontoHistorico'])


#%%
f = Fitter(np.log(datos.MontoHistorico))

f.fit()



#%%


f.summary()

#fit = pareto.fit(datos.MontoHistorico)
#print(fit)




#%%
logeados = np.log(datos.MontoHistorico)

f = Fitter(logeados[logeados <= 14.65],  
           distributions= ['pareto', 'gamma', 'dweibull','gennorm'])

f.fit()

f.summary()





#%%
