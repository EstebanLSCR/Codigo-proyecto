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

































# alpha = nivel de significancia, 0.95, 0.99...
# parametros[0] = "normal", "pareto"...
# parametros[1] = parametro 1
# parametros[2] = parametros 2, ya viene como scale
# ...

parametros = ("normal", 1, 2)
alpha = 0.95

def VaR_alpha(alpha, parametros):
    
    if(parametros[0] == "normal"):
        from scipy.stats import norm
        VaR = norm.ppf(alpha,parametros[1],parametros[2])
    
    elif(parametros[0] == "gamma"):
        from scipy.stats import gamma
        VaR = gamma.ppf(alpha,parametros[1],scale=parametros[2])
    
    elif(parametros[0] == "pareto"):
        from scipy.stats import pareto
        VaR = pareto.ppf(q=alpha,b=parametros[1],scale=parametros[2])
    
    
    
    
    return VaR
    



