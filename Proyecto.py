# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 06:50:33 2020

@author: Amel
@author: Esteban
@author: Marco
"""


# Código proyecto de riesgo

#%%

# Cargando paquetes y los datos
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from fitter import Fitter
import statsmodels.api as sm
import datetime

datos = pd.read_excel("Defraudaciones enero-junio 2020.xlsx")

#%%

#Histograma

plt.hist(datos['MontoHistorico'])
plt.title("Histograma para las reclamaciones")
plt.xlabel('Reclamos')
plt.ylabel('Conteo') 


#%%

## Ajustando para valores con la transformación logarítmica

logeados = np.log(datos.MontoHistorico)


f = Fitter(logeados,  
           distributions= ['pareto', 'gamma', 'dweibull',
                           'gennorm'])

f.fit()
f.summary()

plt.title("Ajuste de densidades para las reclamaciones")
plt.xlabel('Reclamos transformación logarítmica')
plt.ylabel('Densidad') 


# Para salvar imagen
## plt.savefig('Prueba.svg', format='svg', dpi=1200)


#%%

## Gráfico de cuantiles

parametros_pareto = f.fitted_param['pareto']
parametros_normal = f.fitted_param['gennorm']
parametros_weibull = f.fitted_param['dweibull']
parametros_gamma = f.fitted_param['gamma']

fig = plt.figure()

ax = fig.add_subplot(2, 2, 1)
sm.qqplot(logeados, stats.gennorm, 
          distargs= (parametros_normal[0],) , 
          loc = parametros_normal[1], 
          scale = parametros_normal[2],
          line = "45", ax = ax)
ax.set_title('Normal', size = 11.0)
ax.set_xlabel("")
ax.set_ylabel("")

ax2 = fig.add_subplot(2, 2, 2)
sm.qqplot(logeados, stats.pareto, 
        distargs= (parametros_pareto[0],) , 
        loc = parametros_pareto[1], 
        scale = parametros_pareto[2],
          line = "45",ax = ax2)
ax2.set_title('Pareto', size = 11.0)
ax2.set_xlabel("")
ax2.set_ylabel("")

ax3 = fig.add_subplot(2, 2, 3)
sm.qqplot(logeados, stats.dweibull, 
          distargs= (parametros_weibull[0],) ,
          loc = parametros_weibull[1], 
          scale = parametros_weibull[2],
          line = "45", ax = ax3)
ax3.set_title('Weibull', size = 11.0)
ax3.set_xlabel("")
ax3.set_ylabel("")


ax4 = fig.add_subplot(2, 2, 4)
sm.qqplot(logeados, stats.gamma, 
        distargs= (parametros_gamma[0],) ,
        loc = parametros_gamma[1], 
        scale = parametros_gamma[2],
          line = "45", ax = ax4)
ax4.set_title('Gamma', size = 11.0)
ax4.set_xlabel("")
ax4.set_ylabel("")

fig.tight_layout(pad=0.7)

fig.text(0.5, 0, 'Cuantiles teoricos', ha='center', va='center')
fig.text(0., 0.5, 'Cuantiles observados', ha='center', va='center', rotation='vertical')

fig.suptitle('Gráfico de cuantiles distribuciones ajustadas')
fig.subplots_adjust(top=0.86)

plt.show()

#%%

fechas = np.array([datetime.datetime.strptime(str(datos.FechaDescubrimiento[0]), 
                               "%Y-%m-%d %H:%M:%S") ])
for i in range(0, len(datos.FechaDescubrimiento)):
    temp = datetime.datetime.strptime(str(datos.FechaDescubrimiento[i]), 
                               "%Y-%m-%d %H:%M:%S")
    fechas = np.append(fechas, temp)

meses = [x.month for x in fechas]


#%%


plt.hist(meses, bins = 12)

plt.title("Histograma de meses de descubrimiento")
plt.xlabel('Mes')
plt.ylabel('Conteo') 







#%%





# alpha = nivel de significancia, 0.95, 0.99...
# parametros[0] = "normal", "pareto"...
# parametros[1] = parametro 1
# parametros[2] = parametros 2, ya viene como scale
# ...

parametros = ("normal", 1, 2)
alpha = 0.95

def VaR_alpha(alpha, parametros):
    
    if(parametros[0] == "gennormal"):
        from scipy.stats import gennorm
        VaR = gennorm.ppf(alpha,parametros[1],parametros[2])
        
    elif(parametros[0] == "normal"):
        from scipy.stats import norm
        VaR = norm.ppf(alpha,parametros[1],parametros[2])
    
    elif(parametros[0] == "gamma"):
        from scipy.stats import gamma
        VaR = gamma.ppf(alpha,parametros[1],scale=parametros[2])
    
    elif(parametros[0] == "pareto"):
        from scipy.stats import pareto
        VaR = pareto.ppf(q=alpha,b=parametros[1],scale=parametros[2])
    
    elif(parametros[0] == "weibull"):
        from scipy.stats import weibull
        VaR = weibull.ppf(q=alpha,b=parametros[1],scale=parametros[2])
    
    else: #(parametros[0] == "lognorm"):
        from scipy.stats import lognorm
        VaR = lognorm.ppf(q=alpha,b=parametros[1],scale=parametros[2])
        
    return VaR
    



