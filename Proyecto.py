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
from datetime import datetime, timedelta
# from datetime import timedelta
# from scipy.stats import chisquare
# import datetime
import seaborn as sns
import locale

datos = pd.read_excel("Defraudaciones enero-junio 2020.xlsx")

#%%

#Histograma

# plt.hist(datos['MontoHistorico'])
# plt.title("Histograma para las reclamaciones")
# plt.xlabel('Reclamos')
# plt.ylabel('Conteo') 


#%%

logeados = np.log(datos.MontoHistorico)

f = Fitter(logeados, distributions= ['gamma', 'dweibull', 'gennorm'])

f.fit()
f.summary()

parametros_pareto = stats.genpareto.fit(logeados, loc = 2 )

f_pareto =  stats.genpareto.pdf(np.sort(logeados), 
                c = parametros_pareto[0], 
                loc = parametros_pareto[1], 
                scale = parametros_pareto[2])


plt.plot(np.sort(logeados), f_pareto, color = "purple", label = "genpareto")
plt.legend()


plt.title("Ajuste de densidades para las reclamaciones")
plt.xlabel('Reclamos transformación logarítmica')
plt.ylabel('Densidad') 


# Para salvar imagen
## plt.savefig('Prueba.svg', format='svg', dpi=1200)

# plt.savefig('Densidad.jpeg', format='jpeg', dpi=1300)


#%% Frecuencias

#datos['mesO']=0
#datos['mesD']=0
# for i in range(0,len(datos.MontoHistorico)):
#     datos.mesO[i]=(datos.FechaOcurrencia[i]).month
#     datos.mesR[i]=(datos.FechaRegistro[i]).month
# datos['mesO']=0
# datos['mesR']=0
# for i in range(0,len(datos.MontoHistorico)):
#     datos.mesO[i]=(datos.FechaOcurrencia[i]).month
#     datos.mesR[i]=(datos.FechaRegistro[i]).month

    
# #datos 
# #tipo: "O", "D"
# #...

# def F_frecuencias(datos):
#     #frec=[1,2,3,4,5,6,7,8,9,10,11,12]
#     frecO=datos["MontoHistorico"]
#     frecO=frecO[1:12]
#     #frecD=[1,2,3,4,5,6,7,8,9,10,11,12]
    
    
#     for i in range(1,13): 
#             frecO[i] = len((datos>>
#                                mask(X.mesO==i)).mesO)
#for i in range(1,13): 
#    frecO[i] = len((datos.mask(X.mesR==i)).mesO)
            
#             #frecD[i] = len((datos>>
#                             #mask(X.mesD==i)).mesD)
            
#    # if(tipo=="O"):
#    #            frec = frecO
                
#    # elif(tipo=="D"):
#    #             frec = frecd
            
                
#     return frecO


#%%

## Gráfico de cuantiles

parametros_pareto = stats.genpareto.fit(logeados, loc = 2 )
parametros_normal = f.fitted_param['gennorm']
parametros_weibull = f.fitted_param['dweibull']
parametros_gamma = f.fitted_param['gamma']

fig = plt.figure(dpi = 1300)

ax = fig.add_subplot(2, 2, 1)
sm.qqplot(logeados, stats.gennorm, 
          distargs= (parametros_normal[0],) , 
          loc = parametros_normal[1], 
          scale = parametros_normal[2],
          line = "45", ax = ax)
ax.set_title('Normal generalizada', size = 11.0)
ax.set_xlabel("")
ax.set_ylabel("")
#ax.set_xlim([3, 20])
#ax.set_ylim([3, 20])


ax2 = fig.add_subplot(2, 2, 2)
sm.qqplot(logeados, stats.genpareto, 
        distargs= (parametros_pareto[0],) , 
        loc = parametros_pareto[1], 
        scale = parametros_pareto[2],
          line = "45",ax = ax2)
ax2.set_title('Pareto generalizada', size = 11.0)
ax2.set_xlabel("")
ax2.set_ylabel("")
#ax2.set_xlim([3, 20])
#ax2.set_ylim([3, 20])


ax3 = fig.add_subplot(2, 2, 3)
sm.qqplot(logeados, stats.dweibull, 
          distargs= (parametros_weibull[0],) ,
          loc = parametros_weibull[1], 
          scale = parametros_weibull[2],
          line = "45", ax = ax3)
ax3.set_title('Weibull doble', size = 11.0)
ax3.set_xlabel("")
ax3.set_ylabel("")
#ax3.set_xlim([3, 20])
#ax3.set_ylim([3, 20])


ax4 = fig.add_subplot(2, 2, 4)
sm.qqplot(logeados, stats.gamma, 
        distargs= (parametros_gamma[0],) ,
        loc = parametros_gamma[1], 
        scale = parametros_gamma[2],
          line = "45", ax = ax4)
ax4.set_title('Gamma', size = 11.0)
ax4.set_xlabel("")
ax4.set_ylabel("")
#ax4.set_xlim([3, 20])
#ax4.set_ylim([3, 20])

fig.tight_layout(pad=0.7)

fig.text(0.5, 0, 'Cuantiles teóricos', ha='center', va='center')
fig.text(0., 0.5, 'Cuantiles observados', ha='center', va='center', rotation='vertical')

fig.suptitle('Gráfico de cuantiles distribuciones ajustadas')
fig.subplots_adjust(top=0.86)
plt.show()

#%%

## Gráfico pp distribución completa


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



fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2, 
                                          dpi=1300)


pp_plot(logeados, stats.gennorm(beta = parametros_normal[0], 
                                loc = parametros_normal[1],
                                scale=parametros_normal[2]), 
        line = True, ax=ax1)

ax1.set_title('Normal generalizada', fontsize=11)


pp_plot(logeados, stats.genpareto(c = parametros_pareto[0], 
                                loc = parametros_pareto[1],
                                scale=parametros_pareto[2]), 
        line = True,ax=ax2)
ax2.set_title('Pareto generalizada', fontsize=11)

pp_plot(logeados, stats.dweibull(c = parametros_weibull[0], 
                                loc = parametros_weibull[1],
                                scale=parametros_weibull[2]), 
        line = True,ax=ax3)
ax3.set_title('Weibull doble', fontsize=11)

pp_plot(logeados, stats.gamma(a = parametros_gamma[0], 
                                loc = parametros_gamma[1],
                                scale=parametros_gamma[2]), 
        line = True,ax=ax4)
ax4.set_title('Gamma', fontsize=11)

fig.tight_layout(pad=0.7)

fig.text(0.5, 0, 'Probabilidades teóricas', ha='center', va='center')
fig.text(0., 0.5, 'Probabilidades observadas', ha='center', va='center', rotation='vertical')

fig.suptitle('Gráfico de probabilidades observadas vs teóricas')
fig.subplots_adjust(top=0.86)

plt.show()

#%%

# fechas = np.array([datetime.datetime.strptime(str(datos.FechaDescubrimiento[0]), 
#                                "%Y-%m-%d %H:%M:%S") ])

# for i in range(1, len(datos.FechaDescubrimiento)):
#     temp = datetime.datetime.strptime(str(datos.FechaDescubrimiento[i]), 
#                                "%Y-%m-%d %H:%M:%S")
#     fechas = np.append(fechas, temp)

# meses = [x.month-1 for x in fechas]

#df=datos.loc[:,["MontoHistorico","frecuenciaO"]]
#df["BigFrec"]=df["frecuenciaO"]>30
#observados=pd.crosstab(index=df['BigFrec'],columns=df['MontoHistorico'],margins=True)

#res = stat()

#res.chisq(df=datos)

#Prueba chi cuadrado
#vec=(F_frecuencias(datos,"O"))
#chisquare(vec)


#%%

#Corte para valor extremo

logeados2 = logeados[ logeados >= np.quantile(logeados, 0.95)]

f2 = Fitter(logeados2,  
           distributions= ['gamma', 'dweibull',
                           'gennorm'])
f2.fit()

plt.title("Histograma de meses de descubrimiento")
plt.xlabel('Mes')
plt.ylabel('Conteo') 


parametros_pareto2 = stats.genpareto.fit(logeados2, loc = 15 )

f_pareto =  stats.genpareto.pdf(np.sort(logeados2), 
                c = parametros_pareto2[0], 
                loc = parametros_pareto2[1], 
                scale = parametros_pareto2[2])

parametros_normal2 = f2.fitted_param['gennorm']
parametros_weibull2 = f2.fitted_param['dweibull']
parametros_gamma2 = f2.fitted_param['gamma']

f2.summary()
plt.plot(np.sort(logeados2), f_pareto, color = "purple", label = "genpareto")
plt.legend()


plt.title("Ajuste de densidades para la cola de las reclamaciones")
plt.xlabel('Reclamos transformación logarítmica')
plt.ylabel('Densidad') 

plt.savefig('Densidad_cola.jpeg', format='jpeg', dpi=1300)

plt.show()

#%%

## Gráfico de cuantiles cola

fig = plt.figure(dpi = 1300)

ax = fig.add_subplot(2, 2, 1)
sm.qqplot(logeados2, stats.gennorm, 
          distargs= (parametros_normal2[0],) , 
          loc = parametros_normal2[1], 
          scale = parametros_normal2[2],
          line = "45", ax = ax)
ax.set_title('Normal generalizada', size = 11.0)
ax.set_xlabel("")
ax.set_ylabel("")
#ax.set_xlim([12, 17])
#ax.set_ylim([12, 17])


ax2 = fig.add_subplot(2, 2, 2)
sm.qqplot(logeados2, stats.genpareto, 
        distargs= (parametros_pareto2[0],) , 
        loc = parametros_pareto2[1], 
        scale = parametros_pareto2[2],
          line = "45",ax = ax2)
ax2.set_title("Pareto generalizada", size = 11.0)
ax2.set_xlabel("")
ax2.set_ylabel("")
#ax2.set_xlim([12, 20])
#ax2.set_ylim([12, 20])


ax3 = fig.add_subplot(2, 2, 3)
sm.qqplot(logeados2, stats.dweibull, 
          distargs= (parametros_weibull2[0],) ,
          loc = parametros_weibull2[1], 
          scale = parametros_weibull2[2],
          line = "45", ax = ax3)
ax3.set_title('Weibull doble', size = 11.0)
ax3.set_xlabel("")
ax3.set_ylabel("")
#ax3.set_xlim([12, 20])
#ax3.set_ylim([12, 20])


ax4 = fig.add_subplot(2, 2, 4)
sm.qqplot(logeados2, stats.gamma, 
        distargs= (parametros_gamma2[0],) ,
        loc = parametros_gamma2[1], 
        scale = parametros_gamma2[2],
          line = "45", ax = ax4)
ax4.set_title('Gamma', size = 11.0)
ax4.set_xlabel("")
ax4.set_ylabel("")
#ax4.set_xlim([12, 20])
#ax4.set_ylim([12, 20])

fig.tight_layout(pad=0.7)

fig.text(0.5, 0, 'Cuantiles teóricos', ha='center', va='center')
fig.text(0., 0.5, 'Cuantiles observados', ha='center', va='center', rotation='vertical')

fig.suptitle('Gráfico de cuantiles distribuciones para las colas')
fig.subplots_adjust(top=0.86)

plt.show()

#%%
mes = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "setiembre", "octubre", "noviembre", "diciembre"]

fechas = np.array([datetime.strptime(str(datos.FechaRegistro[0]), "%Y-%m-%d %H:%M:%S") ])
for i in range(1, len(datos.FechaRegistro)):
    temp = datetime.strptime(str(datos.FechaRegistro[i]), "%Y-%m-%d %H:%M:%S")
    fechas = np.append(fechas, temp)

meses = [mes[x.month-1] for x in fechas]



#%%

sns.countplot(meses)
plt.title("Histograma de meses de registro")
plt.xlabel('Meses')
plt.ylabel('Conteo') 

plt.savefig('Conteo_meses_registro.jpeg', format='jpeg', dpi=1300)
            

#%% Frecuencias

fechas = datos.FechaRegistro
inicio = datetime(2020,1,1)
fin    = datetime(2020,6,30)

datosF = pd.DataFrame({'Fechas' : datos.FechaRegistro})

ceros = np.arange((fin - inicio).days + 1 - datosF.nunique(0).Fechas)*0

f = np.array(datosF.Fechas.value_counts())

frecuencias = np.concatenate((f,ceros))

n = len(frecuencias)
m = np.mean(frecuencias)
v = (n - 1)/n * np.var(frecuencias, ddof=1)
size = (m**2/(v - m))*365
p = size/(size+m)



#%% Conteo de reclamos por día

rango_fechas = pd.date_range(datos['FechaRegistro'][0] - timedelta(days = 1), 
        end = datos['FechaRegistro'].max() + timedelta(days = 4) ).to_pydatetime().tolist()


fechas_vistas = np.array([datetime.strptime(str(x), 
                    "%Y-%m-%d %H:%M:%S") for x in datos.FechaRegistro ])


conteo_dias = np.zeros(len(rango_fechas))

for i in range(0,len(rango_fechas)):
    conteo_dias[i] = sum(fechas_vistas == rango_fechas[i])


df = pd.DataFrame({'Fecha': rango_fechas, 'Conteo': conteo_dias })


plt.hist(data=df, x="Conteo", bins = len(conteo_dias));
plt.title("Histograma para el número de reclamos por día")
plt.xlabel('Número de reclamos')
plt.ylabel("Conteo")


#%%

## Parámetros obtenidos en R para la binomial negativa
mu = 4.129051
size = 0.174687
prob = size/(size + mu)

q =  np.linspace(0.01,0.99,182)
cuantil_teorico = stats.nbinom.ppf(q, n = size, p = prob)
cuantil_observado = np.quantile(conteo_dias, q)



fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2)

ax1.scatter(cuantil_teorico, cuantil_observado, color = "blue")
ax1.plot([0, 50], [0, 50], color = "red")
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

plt.savefig('Frecuencias_nbinom.jpeg', format='jpeg', dpi=1300)

plt.show()

#%%

## Parámetros obtenidos en R para la geometrica
p = 0.1948608


q =  np.linspace(0.01,0.99,182)
cuantil_teorico = stats.geom.ppf(q , p)
cuantil_observado = np.quantile(conteo_dias, q)



fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2)

ax1.scatter(cuantil_teorico, cuantil_observado, color = "blue")
ax1.plot([0, 50], [0, 50], color = "red")
ax1.set_title("Cuantiles", fontsize=11)
ax1.set_xlabel('Cuantiles teóricos', fontsize=8)
ax1.set_ylabel("Cuantiles observados", fontsize=8)


cdf_teorico = stats.geom.cdf(cuantil_teorico, p = p)
ecdf = sm.distributions.ECDF(cuantil_observado)

ax2.step(cuantil_teorico, cdf_teorico, label = "CDF teórico",
         color = "red")
ax2.step(cuantil_teorico, ecdf(cuantil_teorico),
            color = "blue",  label = "CDF observado")
ax2.set_title("Distribución acumulada", fontsize=11)
ax2.set_xlabel('Datos', fontsize=8)
ax2.set_ylabel("CDF", fontsize=8)
ax2.legend()


pp_plot(conteo_dias , stats.geom(p=p) , ax = ax3)
ax3.set_title("Probabilidades", fontsize=11)
ax3.set_xlabel('Teóricas', fontsize=8)
ax3.set_ylabel("Observadas", fontsize=8)
       

densidad_teorica = stats.geom.pmf(np.unique(conteo_dias), p = p)

ax4.hist(data=df, x="Conteo", density=True, stacked = True, bins = 45,
         label = "Observada")
ax4.plot(np.unique(conteo_dias), densidad_teorica, color = "red",
         label = "Teórica")
ax4.set_title("Densidad", fontsize=11)
ax4.set_xlabel('Datos', fontsize=8)
ax4.set_ylabel("Densidad", fontsize=8)
ax4.legend()    

fig.suptitle("Ajuste geométrica", fontsize=14)
fig.tight_layout(pad=0.9)
fig.subplots_adjust(top=0.85)

plt.savefig('Frecuencias_geometrica.jpeg', format='jpeg', dpi=1300)

plt.show()

#%%



## Parámetros obtenidos para la poisson
lamb = np.mean(conteo_dias)

q =  np.linspace(0.01,0.99,182)
cuantil_teorico = stats.poisson.ppf(q, mu = lamb)
cuantil_observado = np.quantile(conteo_dias, q)


fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2)

ax1.scatter(cuantil_teorico, cuantil_observado, color = "blue")
ax1.plot([0, 50], [0, 50], color = "red")
ax1.set_title("Cuantiles", fontsize=11)
ax1.set_xlabel('Cuantiles teóricos', fontsize=8)
ax1.set_ylabel("Cuantiles observados", fontsize=8)


cdf_teorico = stats.poisson.cdf(cuantil_teorico, mu = lamb )
ecdf = sm.distributions.ECDF(cuantil_observado)

ax2.step(cuantil_teorico, cdf_teorico, label = "CDF teórico",
         color = "red")
ax2.step(cuantil_teorico, ecdf(cuantil_teorico),
            color = "blue",  label = "CDF observado")
ax2.set_title("Distribución acumulada", fontsize=11)
ax2.set_xlabel('Datos', fontsize=8)
ax2.set_ylabel("CDF", fontsize=8)
ax2.legend()

pp_plot(conteo_dias , stats.poisson(mu = lamb) , ax = ax3)
ax3.set_title("Probabilidades", fontsize=11)
ax3.set_xlabel('Teóricas', fontsize=8)
ax3.set_ylabel("Observadas", fontsize=8)
       

densidad_teorica = stats.poisson.pmf(np.unique(conteo_dias), mu = lamb)

ax4.hist(data=df, x="Conteo", density=True, stacked = True, bins = 45,
         label = "Observada")
ax4.plot(np.unique(conteo_dias), densidad_teorica, color = "red",
         label = "Teórica")
ax4.set_title("Densidad", fontsize=11)
ax4.set_xlabel('Datos', fontsize=8)
ax4.set_ylabel("Densidad", fontsize=8)
ax4.legend()    


fig.suptitle("Ajuste Poisson", fontsize=14)

fig.tight_layout(pad=0.9)
fig.subplots_adjust(top=0.85)

plt.savefig('Frecuencias_poisson.jpeg', format='jpeg', dpi=1300)

plt.show()


#%%

## Pruebas ks
#stats.kstest(conteo_dias, "poisson", args = (lamb,))
#stats.kstest(conteo_dias, "nbinom", args=(size, prob ))

#%%

logeados2 = logeados[ logeados >= np.quantile(logeados, 0.95)]
f2 = Fitter(np.exp(logeados2),  distributions= 'genpareto')
f2.fit()
parametros_pareto2 =  f2.fitted_param['genpareto']


#%%

#### Algoritmo para las simulaciones todo en escala logarítmica

# m: numero de simulaciones
m = 10000
    # Umbral para teoría del valor extremo
q = 0.95
# parámetros binomial negativa

mu = 4.129051
size = 0.174687
prob = size/(size + mu)
#size = 0.2730495
size = size*365

parametros_nbinom = np.array([size,prob])

# Función para sampleo de la distribución empírica

def sampleo_distr_empirica(n):
    ''' 
    n = tamaño del sampleo
    '''
    uniformes = np.random.uniform(size = n)
    sampleo = np.quantile(np.exp(logeados[logeados < np.quantile(logeados, q)]), 
                          uniformes)
    return sampleo
    

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

#%%
logeados

#Media
locale.format_string("%d", np.mean(totales) , grouping=True)

# Var 99
Var = np.quantile(totales, q = 0.99)
locale.format_string("%d", Var , grouping=True)

# ES 99
ES =  np.mean(totales[totales > Var])
locale.format_string("%d",ES, grouping=True)


#%%

# Gráfico histograma de totales
plt.hist(np.log(totales))
plt.title("Histograma de los totales simulados")
plt.xlabel('Logaritmo de los totales')
plt.ylabel('Conteo') 


plt.savefig('hist_de los totales simulados.jpeg', format='jpeg', dpi=1300)

plt.show()





 #%% Pruebas KS

stats.kstest(logeados, "genpareto", args=(parametros_pareto))
stats.kstest(logeados, "gennorm", args=(parametros_normal))
stats.kstest(logeados, "gamma", args=(parametros_gamma))
stats.kstest(logeados, "dweibull", args=(parametros_weibull))

    
stats.kstest(logeados2, "genpareto", args=(parametros_pareto2))
stats.kstest(logeados2, "gennorm", args=(parametros_normal2))
stats.kstest(logeados2, "gamma", args=(parametros_gamma2))
stats.kstest(logeados2, "dweibull", args=(parametros_weibull2))





