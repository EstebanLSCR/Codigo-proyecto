#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:55:57 2020

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
import locale

datos = pd.read_excel("Defraudaciones enero-junio 2020.xlsx")


#%% Frecuencias
fechas = datos.FechaRegistro
inicio = datetime(2020,1,1)
fin    = datetime(2020,6,30)

datosF = pd.DataFrame({'Fechas' : datos.FechaRegistro})
ceros = np.arange((fin - inicio).days + 1 - datosF.nunique(0).Fechas)*0
f = np.array(datosF.Fechas.value_counts())
frecuencias = np.concatenate((f,ceros))


#%%

logeados = np.log(datos.MontoHistorico)
logeados2 = logeados[ logeados >= np.quantile(logeados, 0.95)]
f2 = Fitter(np.exp(logeados2),  distributions= 'genpareto')
f2.fit()
parametros_pareto2 =  f2.fitted_param['genpareto']



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
#### Algoritmo para las simulaciones todo en escala logarítmica usando MLE

# m: numero de simulaciones
m = 10000

# Umbral para teoría del valor extremo
q = 0.95

# Nivel de confianza
a = 0.99


# Código para instalar y correr paquetes de R: fitdistrplus y MASS
import rpy2.robjects.packages as rpackages

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1)

packnames = ('fitdistrplus', 'MASS')

# paquete para crear strings en R
from rpy2.robjects.vectors import StrVector

# Instala paquetes que no estén instalados
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


    
# vector de totales
totales = np.zeros(m) 

# Simulaciones
# Se fija la semilla para obtener replicar los resultados 
np.random.seed(100)

# Genere vector de variables N_1 , ... , N_m
Frecuencias = stats.nbinom.rvs(n = parametros_nbinom[0] , 
                               p = parametros_nbinom[1],
                               loc = 0, size = m)

#maximos = np.zeros(m) 
#no_empiricas = np.zeros(m) 
for j in range(0,m):
    # Genere vector de variables U_N_j,1 , ... , U_N_j,N_j
    Uniformes = np.random.uniform(size = Frecuencias[j])
    # Vector de reclamaciones
    Reclamaciones = np.zeros(Frecuencias[j])
    
    # Reclamaciones con distribución empírica
    empirica = Uniformes < q
    
    # Sampleo reclamaciones según distribución empírica
    Reclamaciones[empirica] = sampleo_distr_empirica(sum(empirica))
    # Sampleo reclamaciones según distribución generalizada de pareto
    Reclamaciones[~ empirica] = stats.genpareto.rvs(c = parametros_pareto2[0],
                        loc = parametros_pareto2[1], scale = parametros_pareto2[2],
                        size = sum(Uniformes > q))
    #maximos[j] = max(Reclamaciones)
    #no_empiricas[j] = sum(Uniformes > q)
    
    # Se guardan la suma de las reclamaciones
    # Elimina el efecto de los logarítmos
    totales[j] = sum(Reclamaciones)

# Var 99
VaR_mle = np.quantile(totales, q = a)

# ES 99
ES_mle =  np.mean(totales[totales > VaR_mle])



#%%

#### Algoritmo para las simulaciones todo en escala logarítmica usando MME

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

# vector de totales
totales = np.zeros(m) 

# Simulaciones
# Se fija la semilla para obtener replicar los resultados 
np.random.seed(100)

# Genere vector de variables N_1 , ... , N_m
Frecuencias = stats.nbinom.rvs(n = parametros_nbinom[0] , 
                               p = parametros_nbinom[1],
                               loc = 0, size = m)

#maximos = np.zeros(m) 
#no_empiricas = np.zeros(m) 
for j in range(0,m):
    # Genere vector de variables U_N_j,1 , ... , U_N_j,N_j
    Uniformes = np.random.uniform(size = Frecuencias[j])
    # Vector de reclamaciones
    Reclamaciones = np.zeros(Frecuencias[j])
    
    # Reclamaciones con distribución empírica
    empirica = Uniformes < q
    
    # Sampleo reclamaciones según distribución empírica
    Reclamaciones[empirica] = sampleo_distr_empirica(sum(empirica))
    # Sampleo reclamaciones según distribución generalizada de pareto
    Reclamaciones[~ empirica] = stats.genpareto.rvs(c = parametros_pareto2[0],
                        loc = parametros_pareto2[1], scale = parametros_pareto2[2],
                        size = sum(Uniformes > q))
    #maximos[j] = max(Reclamaciones)
    #no_empiricas[j] = sum(Uniformes > q)
    
    # Se guardan la suma de las reclamaciones
    # Elimina el efecto de los logarítmos
    totales[j] = sum(Reclamaciones)

# Var 99
VaR_mme = np.quantile(totales, q = a)

# ES 99
ES_mme =  np.mean(totales[totales > VaR_mme])




#%%

# MME
# VaR = 5678947701.201136
# ES = 19011668995.934193
# meantotales = 967842357.6916
# meanFrecuencias = 1508.824

# MLE
# VaR = 5086342086.310953
# ES = 12033669154.629614
# meantotales = 888222219.0142984 
# meanFrecuencias = 1508.2664


# sum(frecuencias)*2 = 1504
# sum(np.exp(logeados))*2 = 828594297.8289995







