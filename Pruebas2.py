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

#%%

datos = datos[datos.TipoEvento=="Tarjetas de crédito"]



#Histograma

plt.hist(datos['MontoHistorico'])
plt.title("Histograma para las reclamaciones")
plt.xlabel('Reclamos')
plt.ylabel('Conteo') 


#%%

logeados = np.log(datos.MontoHistorico)

f = Fitter(logeados, 
           distributions=['dgamma','dweibull','foldcauchy','laplace','loglaplace'])

f.fit()
f.summary()

parametros_genpareto = stats.genpareto.fit(logeados, loc = 2 )

f_pareto =  stats.genpareto.pdf(np.sort(logeados), 
                c = parametros_genpareto[0], 
                loc = parametros_genpareto[1], 
                scale = parametros_genpareto[2])


plt.plot(np.sort(logeados), f_pareto, color = "purple", label = "genpareto")
plt.legend()


plt.title("Ajuste de densidades para las reclamaciones")
plt.xlabel('Reclamos transformación logarítmica')
plt.ylabel('Densidad') 




#%%

## Gráfico de cuantiles

parametros_genpareto = stats.genpareto.fit(logeados, loc = 2 )
parametros_dgamma = f.fitted_param['dgamma']
parametros_dweibull = f.fitted_param['dweibull']
parametros_foldcauchy = f.fitted_param['foldcauchy']
parametros_laplace = f.fitted_param['laplace']
parametros_loglaplace = f.fitted_param['loglaplace']

fig = plt.figure(dpi = 1300)

ax1 = fig.add_subplot(3, 2, 1)
sm.qqplot(logeados, stats.dgamma(parametros_dgamma[0], 
                                 loc = parametros_dgamma[1], 
                                 scale = parametros_dgamma[2]),
          line = "45", ax = ax1)
ax1.set_title('dgamma', size = 11.0)
ax1.set_xlabel("")
ax1.set_ylabel("")

ax2 = fig.add_subplot(3, 2, 2)
sm.qqplot(logeados, stats.genpareto(parametros_genpareto[0], 
                                    loc = parametros_genpareto[1], 
                                    scale = parametros_genpareto[2]),
          line = "45",ax = ax2)
ax2.set_title('Pareto generalizada', size = 11.0)
ax2.set_xlabel("")
ax2.set_ylabel("")

ax3 = fig.add_subplot(3, 2, 3)
sm.qqplot(logeados, stats.dweibull, 
          distargs= (parametros_dweibull[0],) ,
          loc = parametros_dweibull[1], 
          scale = parametros_dweibull[2],
          line = "45", ax = ax3)
ax3.set_title('Weibull doble', size = 11.0)
ax3.set_xlabel("")
ax3.set_ylabel("")

ax4 = fig.add_subplot(3, 2, 4)
sm.qqplot(logeados, stats.foldcauchy(parametros_foldcauchy[0],
                                     parametros_foldcauchy[1],
                                     parametros_foldcauchy[2]),
          line = "45", ax = ax4)
ax4.set_title('foldcauchy', size = 11.0)
ax4.set_xlabel("")
ax4.set_ylabel("")

ax5 = fig.add_subplot(3, 2, 5)
sm.qqplot(logeados, stats.laplace(parametros_laplace[0],
                                  parametros_laplace[1]),
          line = "45", ax = ax5)
ax5.set_title('laplace', size = 11.0)
ax5.set_xlabel("")
ax5.set_ylabel("")

ax6 = fig.add_subplot(3, 2, 6)
sm.qqplot(logeados, stats.loglaplace(parametros_loglaplace[0],
                                    parametros_loglaplace[1],
                                    parametros_loglaplace[2]),
          line = "45", ax = ax6)
ax6.set_title('loglaplace', size = 11.0)
ax6.set_xlabel("")
ax6.set_ylabel("")

fig.tight_layout(pad=0.7)

fig.text(0.5, 0, 'Cuantiles teóricos', ha='center', va='center')
fig.text(0., 0.5, 'Cuantiles observados', ha='center', va='center', rotation='vertical')

fig.suptitle('Gráfico de cuantiles distribuciones ajustadas')
fig.subplots_adjust(top=0.86)
plt.show()


 #%% Pruebas KS

stats.kstest(logeados, "genpareto", args=(parametros_genpareto))
stats.kstest(logeados, "dgamma", args=(parametros_dgamma))
stats.kstest(logeados, "dweibull", args=(parametros_dweibull))
stats.kstest(logeados, "foldcauchy", args=(parametros_foldcauchy))
stats.kstest(logeados, "laplace", args=(parametros_laplace))
stats.kstest(logeados, "loglaplace", args=(parametros_loglaplace))



#%%

#Corte para valor extremo

logeados2 = logeados[ logeados >= np.quantile(logeados, 0.95)]
# len(logeados2)

f2 = Fitter(logeados2, 
            distributions=['dgamma','dweibull','foldcauchy','laplace','loglaplace'])
f2.fit()

plt.title("Histograma de meses de descubrimiento")
plt.xlabel('Mes')
plt.ylabel('Conteo') 


parametros_genpareto2 = stats.genpareto.fit(logeados2, loc = 15 )

f_pareto =  stats.genpareto.pdf(np.sort(logeados2), 
                c = parametros_genpareto2[0], 
                loc = parametros_genpareto2[1], 
                scale = parametros_genpareto2[2])

parametros_dgamma2 = f2.fitted_param['dgamma']
parametros_dweibull2 = f2.fitted_param['dweibull']
parametros_foldcauchy2 = f2.fitted_param['foldcauchy']
parametros_laplace2 = f2.fitted_param['laplace']
parametros_loglaplace2 = f2.fitted_param['loglaplace']



f2.summary()
plt.plot(np.sort(logeados2), f_pareto, color = "purple", label = "genpareto")
plt.legend()


plt.title("Ajuste de densidades para la cola de las reclamaciones")
plt.xlabel('Reclamos transformación logarítmica')
plt.ylabel('Densidad') 

# plt.savefig('Densidad_cola.jpeg', format='jpeg', dpi=1300)

plt.show()

#%%

## Gráfico de cuantiles cola

fig = plt.figure(dpi = 1300)

ax1 = fig.add_subplot(3, 2, 1)
sm.qqplot(logeados2, stats.dgamma(parametros_dgamma2[0], 
                                 loc = parametros_dgamma2[1], 
                                 scale = parametros_dgamma2[2]),
          line = "45", ax = ax1)
ax1.set_title('dgamma', size = 11.0)
ax1.set_xlabel("")
ax1.set_ylabel("")

ax2 = fig.add_subplot(3, 2, 2)
sm.qqplot(logeados2, stats.genpareto(parametros_genpareto2[0], 
                                    loc = parametros_genpareto2[1], 
                                    scale = parametros_genpareto2[2]),
          line = "45",ax = ax2)
ax2.set_title('Pareto generalizada', size = 11.0)
ax2.set_xlabel("")
ax2.set_ylabel("")

ax3 = fig.add_subplot(3, 2, 3)
sm.qqplot(logeados2, stats.dweibull(parametros_dweibull2[0],
                                    loc = parametros_dweibull2[1], 
                                    scale = parametros_dweibull2[2]),
          line = "45", ax = ax3)
ax3.set_title('Weibull doble', size = 11.0)
ax3.set_xlabel("")
ax3.set_ylabel("")

ax4 = fig.add_subplot(3, 2, 4)
sm.qqplot(logeados2, stats.foldcauchy(parametros_foldcauchy2[0],
                                      parametros_foldcauchy2[1],
                                      parametros_foldcauchy2[2]),
          line = "45", ax = ax4)
ax4.set_title('foldcauchy', size = 11.0)
ax4.set_xlabel("")
ax4.set_ylabel("")

ax5 = fig.add_subplot(3, 2, 5)
sm.qqplot(logeados2, stats.laplace(parametros_laplace2[0],
                                parametros_laplace2[1]),
          line = "45", ax = ax5)
ax5.set_title('laplace', size = 11.0)
ax5.set_xlabel("")
ax5.set_ylabel("")

ax6 = fig.add_subplot(3, 2, 6)
sm.qqplot(logeados2, stats.loglaplace(parametros_loglaplace2[0],
                                     parametros_loglaplace2[1],
                                     parametros_loglaplace2[2]),
          line = "45", ax = ax6)
ax6.set_title('loglaplace', size = 11.0)
ax6.set_xlabel("")
ax6.set_ylabel("")

fig.tight_layout(pad=0.7)

fig.text(0.5, 0, 'Cuantiles teóricos', ha='center', va='center')
fig.text(0., 0.5, 'Cuantiles observados', ha='center', va='center', rotation='vertical')

fig.suptitle('Gráfico de cuantiles distribuciones para las colas')
fig.subplots_adjust(top=0.86)

plt.show()



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

# rango_fechas = pd.date_range(datos['FechaRegistro'][0] - timedelta(days = 1), 
#         end = datos['FechaRegistro'].max() + timedelta(days = 5) ).to_pydatetime().tolist()


# fechas_vistas = np.array([datetime.strptime(str(x), 
#                     "%Y-%m-%d %H:%M:%S") for x in datos.FechaRegistro ])


# conteo_dias = np.zeros(len(rango_fechas))

# for i in range(0,len(rango_fechas)):
#     conteo_dias[i] = sum(fechas_vistas == rango_fechas[i])


# df = pd.DataFrame({'Fecha': rango_fechas, 'Conteo': conteo_dias })



#%%

# ## Parámetros obtenidos en R para la binomial negativa
# n = len(conteo_dias)
# mu = np.mean(conteo_dias)
# v = (n - 1)/n * np.var(conteo_dias, ddof=1)
# size = (mu**2/(v - mu))
# prob = size/(size+mu)


# q =  np.linspace(0.01,0.99,182)
# cuantil_teorico = stats.nbinom.ppf(q, n = size, p = prob)
# cuantil_observado = np.quantile(conteo_dias, q)



# fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2)


# sm.qqplot(conteo_dias, stats.nbinom(n = size, p = prob),
#           loc = 0, scale = 1,
#           line = "45", ax = ax1)
# ax1.set_title("Cuantiles", fontsize=11)
# ax1.set_xlabel('Cuantiles teóricos', fontsize=8)
# ax1.set_ylabel("Cuantiles observados", fontsize=8)


# cdf_teorico = stats.nbinom.cdf(cuantil_teorico, n = size, p = prob)
# ecdf = sm.distributions.ECDF(cuantil_observado)

# ax2.step(cuantil_teorico, cdf_teorico, label = "CDF teórico",
#          color = "red")
# ax2.step(cuantil_teorico, ecdf(cuantil_teorico),
#             color = "blue",  label = "CDF observado")
# ax2.set_title("Distribución acumulada", fontsize=11)
# ax2.set_xlabel('Datos', fontsize=8)
# ax2.set_ylabel("CDF", fontsize=8)
# ax2.legend()


# pp_plot(conteo_dias , stats.nbinom(n = size, p=prob) , ax = ax3)
# ax3.set_title("Probabilidades", fontsize=11)
# ax3.set_xlabel('Teóricas', fontsize=8)
# ax3.set_ylabel("Observadas", fontsize=8)
       

# densidad_teorica = stats.nbinom.pmf(np.unique(conteo_dias), n = size, p = prob)

# ax4.hist(data=df, x="Conteo", density=True, stacked = True, bins = 45,
#          label = "Observada")
# ax4.plot(np.unique(conteo_dias), densidad_teorica, color = "red",
#          label = "Teórica")
# ax4.set_title("Densidad", fontsize=11)
# ax4.set_xlabel('Datos', fontsize=8)
# ax4.set_ylabel("Densidad", fontsize=8)
# ax4.legend()    

# fig.suptitle("Ajuste binomial negativa", fontsize=14)
# fig.tight_layout(pad=0.9)
# fig.subplots_adjust(top=0.85)

# # plt.savefig('Frecuencias_nbinom.jpeg', format='jpeg', dpi=1300)

# plt.show()



#%% Frecuencias


fechas = datos.FechaRegistro
inicio = datetime(2020,1,1)
fin    = datetime(2020,6,30)

datosF = pd.DataFrame({'Fechas' : datos.FechaRegistro})

ceros = np.arange((fin - inicio).days + 1 - datosF.nunique(0).Fechas)*0

f = np.array(datosF.Fechas.value_counts())

frecuencias = np.concatenate((f,ceros))


#%%

logeados2 = logeados[ logeados >= np.quantile(logeados, 0.95)]
datos_sampleo = datos.MontoHistorico[datos.MontoHistorico >= np.quantile(datos.MontoHistorico, 0.95)]
f2 = Fitter(datos_sampleo,  distributions= 'genpareto')
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


# # Código para instalar y correr paquetes de R: fitdistrplus y MASS
#import rpy2.robjects.packages as rpackages

# # import R's utility package
# utils = rpackages.importr('utils')

# # select a mirror for R packages
# utils.chooseCRANmirror(ind=1)

# packnames = ('fitdistrplus', 'MASS', 'stats')

# # paquete para crear strings en R
# from rpy2.robjects.vectors import StrVector

# # Instala paquetes que no están instalados
# names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install))

# paquetes para importar y crear vectores
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# # import R's "base" package
# fitdistrplus = importr('fitdistrplus')
# MASS = importr('MASS')
# Stats = importr('stats')

# ajuste_nbinom = fitdistrplus.fitdist(robjects.IntVector(frecuencias),
#                                       "nbinom", "mle")

# # Parámetros binomial negativa 
# size = ajuste_nbinom[0][0]
# mu = ajuste_nbinom[0][1]
# prob = size/(size + mu)
# size = size*365

# parametros_nbinom = np.array([size,prob])
parametros_nbinom = np.array([12.93532,0.06113912])

    
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
    
    # Se guardan la suma de las reclamaciones
    # Elimina el efecto de los logarítmos
    totales[j] = sum(Reclamaciones)

# Var 99
VaR_mle = np.quantile(totales, q = a)

# ES 99
ES_mle =  np.mean(totales[totales > VaR_mle])


#%%

#Simulaciones sin valor extremo usando MLE genpareto

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
    
    Reclamaciones = stats.genpareto.rvs(parametros_genpareto[0],
                                         parametros_genpareto[1],
                                         parametros_genpareto[2],
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
    
    # Se guardan la suma de las reclamaciones
    # Elimina el efecto de los logarítmos
    totales[j] = sum(Reclamaciones)

# Var 99
VaR_mme = np.quantile(totales, q = a)

# ES 99
ES_mme =  np.mean(totales[totales > VaR_mme])



#%%

#Simulaciones sin valor extremo usando MME genpareto

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
    
    Reclamaciones = stats.genpareto.rvs(parametros_genpareto[0],
                                         parametros_genpareto[1],
                                         parametros_genpareto[2],
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


# MLE
# VaR_mle = 93958679.46161568
# ES_mle = 102244813.12066875
# meantotales_mle = 51148650.034233436
# meanFrecuencias_mle = 198.5175

# VaR_mle1 = 160464392.15579706
# ES_mle1 = 174251468.33023345
# meantotales_mle1 = 88451194.13590004
# meanFrecuencias_mle1 = 198.5175

# VaR_mle2 = 177194060.5848579
# ES_mle2 = 302158641.54217803
# meantotales_mle2 = 55495836.579378806
# meanFrecuencias_mle2 = 198.5175




# MME
# VaR_mme = 82601382.65945648
# ES_mme = 88113997.46836632
# meantotales_mme = 51103563.271025755
# meanFrecuencias_mme = 198.6176

# VaR_mme1 = 141645018.20575505
# ES_mme1 = 151332443.37369093
# meantotales_mme1 = 88495392.01280227
# meanFrecuencias_mme1 = 198.6176 

# VaR_mme2 = 166360608.36285147
# ES_mme2 = 276712257.4414988
# meantotales_mme2 = 55660992.20158795
# meanFrecuencias_mme2 = 198.6176



# sum(frecuencias)*2 = 198
# sum(np.exp(logeados))*2 = 52657903.2622




