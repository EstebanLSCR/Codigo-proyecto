# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 06:50:33 2020

@author: Amel
@author: Esteban
@author: Marco
"""


# Código proyecto de riesgo para modelo completo sin filtrar las reclamaciones por tipo
# Se requiere software R para correr y se requiere que R esté en el path de la computadora
# Se requiere que el archivo "Defraudaciones enero-junio 2020.xlsx" esté en el mismo directorio
# Par ejecutar el script dar clic en la flecha verde de arriba que se llama "Ejecutar archivo"
# El script dura varios minutos para completar

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
# Paquetes de R
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

# Instalando paquetes de R que se usaran
utils = rpackages.importr('utils')

# Mirror para la instalación de paquetes
utils.chooseCRANmirror(ind=1)

packnames = ('fitdistrplus', 'MASS')

# Instalar paquetes que no están instalados
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))


datos = pd.read_excel("Defraudaciones enero-junio 2020.xlsx")

#%%

# Transformación logarítmica
logeados = np.log(datos.MontoHistorico)

#Histograma de las reclamaciones 
fig = plt.figure()

plt.hist(logeados )
plt.title("Histograma para las reclamaciones")
plt.xlabel('Reclamos con transformación logarítmica')
plt.ylabel('Conteo') 

plt.show()

#%% Ajuste de las distribuciones gamma, doble weibull y normal generalizada

f = Fitter(logeados, distributions= ['gamma', 'dweibull', 'gennorm'])

f.fit()
f.summary()

## Ajuste de la distribución pareto generalizada
parametros_pareto = stats.genpareto.fit(logeados, loc = 2 )

f_pareto =  stats.genpareto.pdf(np.sort(logeados), 
                c = parametros_pareto[0], 
                loc = parametros_pareto[1], 
                scale = parametros_pareto[2])

### Gráfico de densidades

plt.plot(np.sort(logeados), f_pareto, color = "purple", label = "genpareto")
plt.legend()


plt.title("Ajuste de densidades para las reclamaciones")
plt.xlabel('Reclamos transformación logarítmica')
plt.ylabel('Densidad') 

plt.show()



#%%

# Se obtienen los parámetros de las distribuciones
parametros_pareto = stats.genpareto.fit(logeados, loc = 2 )
parametros_normal = f.fitted_param['gennorm']
parametros_weibull = f.fitted_param['dweibull']
parametros_gamma = f.fitted_param['gamma']


## Gráfico de cuantiles para las distribuciones ajustadas

fig = plt.figure()

ax = fig.add_subplot(2, 2, 1)
sm.qqplot(logeados, stats.gennorm, 
          distargs= (parametros_normal[0],) , 
          loc = parametros_normal[1], 
          scale = parametros_normal[2],
          line = "45", ax = ax)
ax.set_title('Normal generalizada', size = 11.0)
ax.set_xlabel("")
ax.set_ylabel("")


ax2 = fig.add_subplot(2, 2, 2)
sm.qqplot(logeados, stats.genpareto, 
        distargs= (parametros_pareto[0],) , 
        loc = parametros_pareto[1], 
        scale = parametros_pareto[2],
          line = "45",ax = ax2)
ax2.set_title('Pareto generalizada', size = 11.0)
ax2.set_xlabel("")
ax2.set_ylabel("")


ax3 = fig.add_subplot(2, 2, 3)
sm.qqplot(logeados, stats.dweibull, 
          distargs= (parametros_weibull[0],) ,
          loc = parametros_weibull[1], 
          scale = parametros_weibull[2],
          line = "45", ax = ax3)
ax3.set_title('Weibull doble', size = 11.0)
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

fig.text(0.5, 0, 'Cuantiles teóricos', ha='center', va='center')
fig.text(0., 0.5, 'Cuantiles observados', ha='center', va='center', rotation='vertical')

fig.suptitle('Gráfico de cuantiles distribuciones ajustadas')
fig.subplots_adjust(top=0.86)
plt.show()

#%%

## Función para calcular los gráficos de probabilidades empíricas vs teóricas
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


## Gráfico pp distribución completa

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

#Corte para valor extremo
# q = 0.95

logeados2 = logeados[ logeados >= np.quantile(logeados, 0.95)]

parametros_pareto2 = stats.genpareto.fit(np.exp(logeados2))
parametros_gamma2 =  stats.gamma.fit(np.exp(logeados2))

parametros_gennorm = stats.gennorm.fit(np.exp(logeados2))
parametros_weibull2 = stats.dweibull.fit(np.exp(logeados) )


#%%

## Gráfico de cuantiles para el ajuste de la cola

fig = plt.figure()

ax = fig.add_subplot(2, 2, 1)
sm.qqplot(np.exp(logeados2), stats.gennorm, 
          distargs= (parametros_gennorm[0],) , 
          loc = parametros_gennorm[1], 
          scale = parametros_gennorm[2],
          line = "45", ax = ax)
ax.set_title('Normal generalizada', size = 11.0)
ax.set_xlabel("")
ax.set_ylabel("")

ax2 = fig.add_subplot(2, 2, 2)
sm.qqplot(np.exp(logeados2), stats.genpareto(c =parametros_pareto2[0],
                                     loc = parametros_pareto2[1],
                                     scale = parametros_pareto2[2]), 
          line = "45",ax = ax2)
ax2.set_title("Pareto generalizada", size = 11.0)
ax2.set_xlabel("")
ax2.set_ylabel("")

ax3 = fig.add_subplot(2, 2, 3)
sm.qqplot(np.exp(logeados2), stats.dweibull, 
          distargs= (parametros_weibull2[0],) ,
          loc = parametros_weibull2[1], 
          scale = parametros_weibull2[2],
          line = "45", ax = ax3)
ax3.set_title('Weibull doble', size = 11.0)
ax3.set_xlabel("")
ax3.set_ylabel("")

ax4 = fig.add_subplot(2, 2, 4)
sm.qqplot(np.exp(logeados2), stats.gamma, 
        distargs= (parametros_gamma2[0],) ,
        loc = parametros_gamma2[1], 
        scale = parametros_gamma2[2],
          line = "45", ax = ax4)
ax4.set_title('Gamma', size = 11.0)
ax4.set_xlabel("")
ax4.set_ylabel("")

fig.tight_layout(pad=0.7)

fig.text(0.5, 0, 'Cuantiles teóricos', ha='center', va='center')
fig.text(0., 0.5, 'Cuantiles observados', ha='center', va='center', rotation='vertical')

fig.suptitle('Gráfico de cuantiles distribuciones para las colas')
fig.subplots_adjust(top=0.86)

plt.show()



#%% Pruebas KS del ajuste de colas 

print('Valor p genpareto ' + str(stats.kstest(np.exp(logeados2), "genpareto", args=(parametros_pareto2))[1]))
print('Valor p gamma ' + str(stats.kstest(np.exp(logeados2), "gamma", args=(parametros_gamma2))[1]))
print('Valor p dweibull ' + str(stats.kstest(np.exp(logeados2), "dweibull", args=(parametros_weibull2))[1]))
print('Valor p gennorm ' + str(stats.kstest(np.exp(logeados2), "gennorm", args=(parametros_gennorm))[1]))


#%% Conteo de reclamos por día

rango_fechas = pd.date_range(datos['FechaRegistro'][0] - timedelta(days = 1), 
        end = datos['FechaRegistro'].max() + timedelta(days = 4) ).to_pydatetime().tolist()


fechas_vistas = np.array([datetime.strptime(str(x), 
                    "%Y-%m-%d %H:%M:%S") for x in datos.FechaRegistro ])


conteo_dias = np.zeros(len(rango_fechas))

for i in range(0,len(rango_fechas)):
    conteo_dias[i] = sum(fechas_vistas == rango_fechas[i])


df = pd.DataFrame({'Fecha': rango_fechas, 'Conteo': conteo_dias })



## Gráfico de conteo por días 
fig = plt.figure()

plt.hist(data=df, x="Conteo", bins = len(conteo_dias));
plt.title("Conteo número de reclamos por día para los meses de enero-junio 2020")  
          
plt.xlabel('Número de reclamos por día')

plt.ylabel('Conteo')

plt.savefig('Frecuencia.eps', format='eps', dpi=1300)

plt.show()





#%%

## Parámetros  para el ajuste de la distribución poisson
lamb = np.mean(conteo_dias)

q =  np.linspace(0.01,0.99,182)
cuantil_teorico = stats.poisson.ppf(q, mu = lamb)
cuantil_observado = np.quantile(conteo_dias, q)

## Gráfico de revisión de ajustes para la distribución poisson
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

plt.savefig('Frecuencias_poisson.eps', format='eps', dpi=1300)

plt.show()


#%%

## Parámetros para ajuste de la distribución geométrica

p = 1/(1 + np.mean(conteo_dias))


q =  np.linspace(0.01,0.99,182)
cuantil_teorico = stats.geom.ppf(q , p)
cuantil_observado = np.quantile(conteo_dias, q)

## Gráfico de revisión de ajustes para la distribución geométrica

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


plt.show()


#%%

## Parámetros de ajuste para la distribución binomial negativa obtenidos en R
## usando MLE


# Importando paquetes de R para cálculo de parámetros binomial negativa
fitdistrplus = importr('fitdistrplus')
MASS = importr('MASS')
Stats = importr('stats')

# Cálculo de parámetros binomial negativa en R
ajuste_nbinom = fitdistrplus.fitdist(robjects.IntVector(conteo_dias),
                                     "nbinom", "mle")

mu = ajuste_nbinom[0][1]
size = ajuste_nbinom[0][0]
prob = size/(size + mu)

q =  np.linspace(0.01,0.99,182)
cuantil_teorico = stats.nbinom.ppf(q, n = size, p = prob)
cuantil_observado = np.quantile(conteo_dias, q)


## Gráfico de revisión de ajustes para la distribución binomial negativa

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


#%%

#### Algoritmo para las simulaciones modelo con teoría valoe extremo

# m: numero de simulaciones
m = 10000
    # Umbral para teoría del valor extremo
q = 0.95
# nivel alfa para el VaR y ES
alfa = 0.99

# Se multiplica el parámetro de size de la binomial negativa por 365 para el cálculo de un año
size = size*365

# Se guardan en un vector los parámetros de la distribución binomial negativa
parametros_nbinom = np.array([size,prob])

# Función para sampleo de la distribución empírica

def sampleo_distr_empirica(n):
    ''' 
    Esta función realiza un sampleo de la distribución empírica
    n = tamaño del sampleo
    '''
    uniformes = np.random.uniform(size = n)
    datos_sampleo = datos.MontoHistorico[datos.MontoHistorico < np.quantile(datos.MontoHistorico, q)]
    sampleo = np.quantile(datos_sampleo,uniformes)
    return sampleo
    

# vector de totales
totales_valor_extremo = np.zeros(m) 

# Simulaciones
# Se fija la semilla para obtener replicar los resultados 
np.random.seed(100)

# Genere vector de variables N_1 , ... , N_m
Frecuencias = stats.nbinom.rvs(n = parametros_nbinom[0] , 
                               p = parametros_nbinom[1],
                               loc = 0, size = m)

# Simulaciones
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
    totales_valor_extremo[j] = sum(Reclamaciones)

#%% Resultados obtenidos modelo con teoría valor extremo

locale.setlocale(locale.LC_ALL, 'en_US')

#Media
print("Promedio de las simulaciones modelo teoría valor extremo: " +  locale.format_string("%d", np.mean(totales_valor_extremo) , grouping=True))

# Var alfa
Var = np.quantile(totales_valor_extremo,  alfa)
print("VaR " + str(alfa)+  " de las simulaciones modelo teoría valor extremo: "  + locale.format_string("%d", Var , grouping=True))

# ES alfa
ES =  np.mean(totales_valor_extremo[totales_valor_extremo > Var])
print("ES " + str(alfa)+  " de las simulaciones modelo teoría valor extremo: " + locale.format_string("%d",ES, grouping=True))


#%%

#Ajuste de la distribución normal generalizada a los montos con trasnformación logarítmica

parametros_lognormal = stats.gennorm.fit(logeados)

#%%

#### Algoritmo para las simulaciones modelo con ajuste log normal


# m: numero de simulaciones
m = 10000
# parámetros binomial negativa

parametros_nbinom = np.array([size,prob])


# vector de totales
totales_log_normal = np.zeros(m) 

# Simulaciones
# Se fija la semilla para obtener replicar los resultados 
np.random.seed(100)

# Genere vector de variables N_1 , ... , N_m
Frecuencias = stats.nbinom.rvs(n = parametros_nbinom[0] , 
                               p = parametros_nbinom[1],
                               loc = 0, size = m)


for j in range(0,m):
    # Vector de reclamaciones
    
    Reclamaciones = stats.gennorm.rvs(beta = parametros_lognormal[0],
                        loc = parametros_lognormal[1], 
                        scale = parametros_lognormal[2],
                        size = Frecuencias[j] )
 
    # Se guardan la suma de las reclamaciones
    # Elimina el efecto de los logarítmos
    totales_log_normal[j] = sum(np.exp(Reclamaciones))




#%% Resultados obtenidos modelo con ajuste log-normal

locale.setlocale(locale.LC_ALL, 'en_US')

#Media
print("Promedio de las simulaciones modelo ajuste log-normal: " +  locale.format_string("%d", np.mean(totales_log_normal) , grouping=True))

# Var alfa
Var = np.quantile(totales_log_normal, alfa)
print("VaR " + str(alfa)+  " de las simulaciones modelo ajuste log-normal: "  + locale.format_string("%d", Var , grouping=True))

# ES alfa
ES =  np.mean(totales_log_normal[totales_log_normal > Var])
print("ES " + str(alfa)+  " de las simulaciones modelo ajuste log-normal: " + locale.format_string("%d",ES, grouping=True))





