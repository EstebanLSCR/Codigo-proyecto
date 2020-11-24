#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 06:07:27 2020

@author: marcoantoniomejiaelizondo
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
from datetime import datetime, timedelta
import seaborn as sns


datos = pd.read_excel("Defraudaciones enero-junio 2020.xlsx")
datos = datos[datos.TipoEvento=="Tarjetas de crédito"]


#%%

## Gráfico de densidades
logeados = np.log(datos.MontoHistorico)

f = Fitter(logeados, 
           distributions=['dgamma','dweibull'])

f.fit()
f.summary()

plt.title("Ajuste de densidades para las reclamaciones")
plt.xlabel('Reclamos transformación logarítmica')
plt.ylabel('Densidad') 


#%%

## Gráfico de cuantiles

parametros_dgamma = f.fitted_param['dgamma']
parametros_dweibull = f.fitted_param['dweibull']

fig = plt.figure(dpi = 1300)

ax1 = fig.add_subplot(1, 2, 1)
sm.qqplot(logeados, stats.dgamma(parametros_dgamma[0], 
                                 loc = parametros_dgamma[1], 
                                 scale = parametros_dgamma[2]),
          line = "45", ax = ax1)
ax1.set_title('dgamma', size = 11.0)
ax1.set_xlabel("")
ax1.set_ylabel("")

ax2 = fig.add_subplot(1, 2, 2)
sm.qqplot(logeados, stats.dweibull(parametros_dweibull[0], 
                                    loc = parametros_dweibull[1], 
                                    scale = parametros_dweibull[2]),
          line = "45",ax = ax2)
ax2.set_title('dweibull', size = 11.0)
ax2.set_xlabel("")
ax2.set_ylabel("")

fig.tight_layout(pad=0.7)

fig.text(0.5, 0, 'Cuantiles teóricos', ha='center', va='center')
fig.text(0., 0.5, 'Cuantiles observados', ha='center', va='center', rotation='vertical')

fig.suptitle('Gráfico de cuantiles distribuciones ajustadas')
fig.subplots_adjust(top=0.86)
plt.show()


 #%% Pruebas KS

stats.kstest(logeados, "dgamma", args=(parametros_dgamma))
stats.kstest(logeados, "dweibull", args=(parametros_dweibull))


#%%


def pp_plot(x, dist, line=True, ax=None):
    '''
    Function for comparing empirical data to a theoretical distribution by using a P-P plot.
    
    Params:
    x - empirical data
    dist - distribution object from scipy.stats; for example scipy.stats.norm(0, 1)
    line - boolean; specify if the reference line (y=x) should be drawn on the plot
    ax - specified ax for subplots, None is standalone
    '''
    if ax is None:
        ax = plt.figure().add_subplot(1, 1, 1)
        
    n = len(x)
    p = np.arange(1, n + 1) / n - 0.5 / n
    pp = np.sort(dist.cdf(x))
    
    sns.scatterplot(x=p, y=pp, color='blue', edgecolor='blue', ax=ax,
                    s = 8)
    
    ax.margins(x=0, y=0)
    
    if line: ax.plot(np.linspace(0, 1), np.linspace(0, 1), 'r', lw=2)
    
    return ax



#%% 

rango_fechas = pd.date_range(datos['FechaRegistro'][32] - timedelta(days = 13), 
        end = datos['FechaRegistro'].max() + timedelta(days = 4) ).to_pydatetime().tolist()


fechas_vistas = np.array([datetime.strptime(str(x), 
                    "%Y-%m-%d %H:%M:%S") for x in datos.FechaRegistro ])


conteo_dias = np.zeros(len(rango_fechas))

for i in range(0,len(rango_fechas)):
    conteo_dias[i] = sum(fechas_vistas == rango_fechas[i])


df = pd.DataFrame({'Fecha': rango_fechas, 'Conteo': conteo_dias })



#%%

## Parámetros obtenidos para la binomial negativa
n = len(conteo_dias)
mu = np.mean(conteo_dias)
v = (n - 1)/n * np.var(conteo_dias, ddof=1)
size = (mu**2/(v - mu))
prob = size/(size+mu)


q =  np.linspace(0.01,0.99,182)
cuantil_teorico = stats.nbinom.ppf(q, n = size, p = prob)
cuantil_observado = np.quantile(conteo_dias, q)



fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2)


sm.qqplot(conteo_dias, stats.nbinom(n = size, p = prob),
          loc = 0, scale = 1,
          line = "45", ax = ax1)
ax1.set_title("Cuantiles", fontsize=11)
ax1.set_xlabel('Cuantiles teóricos', fontsize=8)
ax1.set_ylabel("Cuantiles observados", fontsize=8)


cdf_teorico = stats.nbinom.cdf(cuantil_teorico, n = size, p = prob)
ecdf = sm.distributions.ECDF(cuantil_observado)

ax2.step(cuantil_teorico, cdf_teorico, label = "CDF teórico",
          color = "red")
ax2.step(cuantil_teorico, ecdf(cuantil_teorico),
            color = "blue",  label = "CDF observado")
ax2.set_title("Distribución acumulada", fontsize=11)
ax2.set_xlabel('Datos', fontsize=8)
ax2.set_ylabel("CDF", fontsize=8)
ax2.legend()


pp_plot(conteo_dias , stats.nbinom(n = size, p=prob) , ax = ax3)
ax3.set_title("Probabilidades", fontsize=11)
ax3.set_xlabel('Teóricas', fontsize=8)
ax3.set_ylabel("Observadas", fontsize=8)
       

densidad_teorica = stats.nbinom.pmf(np.unique(conteo_dias), n = size, p = prob)

ax4.hist(data=df, x="Conteo", density=True, stacked = True, bins = 45,
          label = "Observada")
ax4.plot(np.unique(conteo_dias), densidad_teorica, color = "red",
          label = "Teórica")
ax4.set_title("Densidad", fontsize=11)
ax4.set_xlabel('Datos', fontsize=8)
ax4.set_ylabel("Densidad", fontsize=8)
ax4.legend()    

fig.suptitle("Ajuste binomial negativa", fontsize=14)
fig.tight_layout(pad=0.9)
fig.subplots_adjust(top=0.85)

plt.show()



#%% Frecuencias


fechas = datos.FechaRegistro
inicio = datetime(2020,1,1)
fin    = datetime(2020,6,30)

datosF = pd.DataFrame({'Fechas' : datos.FechaRegistro})

ceros = np.arange((fin - inicio).days + 1 - datosF.nunique(0).Fechas)*0

f = np.array(datosF.Fechas.value_counts())

frecuencias = np.concatenate((f,ceros))


#%%
# Función para sampleo de la distribución empírica

def sampleo_distr_empirica(n):
    ''' 
    n = tamaño del sampleo
    '''
    uniformes = np.random.uniform(size = n)
    sampleo = np.quantile(np.exp(logeados[logeados < np.quantile(logeados, q)]), 
                          uniformes)
    return sampleo



#%%
## Calculo parametros binomial negativa usando MLE

# m: numero de simulaciones
m = 10000

# Nivel de confianza
a = 0.99


# Código para instalar y correr paquetes de R: fitdistrplus y MASS
import rpy2.robjects.packages as rpackages

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1)

packnames = ('fitdistrplus', 'MASS', 'stats')

# paquete para crear strings en R
from rpy2.robjects.vectors import StrVector

# Instala paquetes que no están instalados
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# paquetes para importar y crear vectores
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# import R's "base" package
fitdistrplus = importr('fitdistrplus')
MASS = importr('MASS')
Stats = importr('stats')

ajuste_nbinom = fitdistrplus.fitdist(robjects.IntVector(frecuencias),
                                      "nbinom", "mle")

# Parámetros binomial negativa 
size = ajuste_nbinom[0][0]
mu = ajuste_nbinom[0][1]
prob = size/(size + mu)
size = size*365

parametros_nbinom = np.array([size,prob])

#%%

#Simulaciones sin valor extremo usando MLE dgamma

# m: numero de simulaciones
m = 10000

# Nivel de confianza
a = 0.99

# Fija la semilla para replicar los resultados
np.random.seed(100)


# vector de totales
totales = np.zeros(m) 


# Genere vector de variables N_1 , ... , N_m
Frecuencias = stats.nbinom.rvs(n = parametros_nbinom[0] , 
                               p = parametros_nbinom[1],
                               loc = 0, size = m)


for j in range(0,m):
    # Vector de reclamaciones
    
    Reclamaciones = stats.dgamma.rvs(parametros_dgamma[0],
                                         parametros_dgamma[1],
                                         parametros_dgamma[2],
                                         size = Frecuencias[j])
    # Elimina el efecto de los logarítmos
    totales[j] = sum(np.exp(Reclamaciones))

# Var 99
VaR_mle1 = np.quantile(totales, q = a)

# ES 99
ES_mle1 =  np.mean(totales[totales > VaR_mle1])



#%%

#Simulaciones sin valor extremo usando MLE dweibull

# m: numero de simulaciones
m = 10000

# Nivel de confianza
a = 0.99

# Fija la semilla para replicar los resultados
np.random.seed(100)


# vector de totales
totales = np.zeros(m) 


# Genere vector de variables N_1 , ... , N_m
Frecuencias = stats.nbinom.rvs(n = parametros_nbinom[0] , 
                               p = parametros_nbinom[1],
                               loc = 0, size = m)


for j in range(0,m):
    # Vector de reclamaciones
    
    Reclamaciones = stats.dweibull.rvs(parametros_dweibull[0],
                                       parametros_dweibull[1],
                                       parametros_dweibull[2],
                                       size = Frecuencias[j])
    # Elimina el efecto de los logarítmos
    totales[j] = sum(np.exp(Reclamaciones))

# Var 99
VaR_mle2 = np.quantile(totales, q = a)

# ES 99
ES_mle2 =  np.mean(totales[totales > VaR_mle2])







#%%

## Calculo parametros binomial negativa usando MME

# m: numero de simulaciones
m = 10000

# Umbral para teoría del valor extremo
q = 0.95

# Nivel de confianza
a = 0.99


# Parámetros binomial negativa 
n = len(frecuencias)
mu = np.mean(frecuencias)
v = (n - 1)/n * np.var(frecuencias, ddof=1)
size = (mu**2/(v - mu))
prob = size/(size+mu)
size = size*365

parametros_nbinom = np.array([size,prob])

#%%

#Simulaciones sin valor extremo usando MME dgamma

# m: numero de simulaciones
m = 10000

# Nivel de confianza
a = 0.99

# Fija la semilla para replicar los resultados
np.random.seed(100)


# vector de totales
totales = np.zeros(m) 


# Genere vector de variables N_1 , ... , N_m
Frecuencias = stats.nbinom.rvs(n = parametros_nbinom[0] , 
                               p = parametros_nbinom[1],
                               loc = 0, size = m)


for j in range(0,m):
    # Vector de reclamaciones
    
    Reclamaciones = stats.dgamma.rvs(parametros_dgamma[0],
                                         parametros_dgamma[1],
                                         parametros_dgamma[2],
                                         size = Frecuencias[j])
    # Elimina el efecto de los logarítmos
    totales[j] = sum(np.exp(Reclamaciones))

#Promedio
promedio_mme = np.mean(totales)

# Var 99
VaR_mme1 = np.quantile(totales, q = a)

# ES 99
ES_mme1 =  np.mean(totales[totales > VaR_mme1])






#%%

#Simulaciones sin valor extremo usando MME dweibull

# m: numero de simulaciones
m = 10000

# Nivel de confianza
a = 0.99

# Fija la semilla para replicar los resultados
np.random.seed(100)


# vector de totales
totales = np.zeros(m) 


# Genere vector de variables N_1 , ... , N_m
Frecuencias = stats.nbinom.rvs(n = parametros_nbinom[0] , 
                               p = parametros_nbinom[1],
                               loc = 0, size = m)


for j in range(0,m):
    # Vector de reclamaciones
    
    Reclamaciones = stats.dweibull.rvs(parametros_dweibull[0],
                                       parametros_dweibull[1],
                                       parametros_dweibull[2],
                                       size = Frecuencias[j])
    # Elimina el efecto de los logarítmos
    totales[j] = sum(np.exp(Reclamaciones))


#Promedio
promedio_mme = np.mean(totales)

# Var 99
VaR_mme2 = np.quantile(totales, q = a)

# ES 99
ES_mme2 =  np.mean(totales[totales > VaR_mme2])





#%%


