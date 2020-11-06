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
import seaborn as sns

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
           distributions= ['gamma', 'dweibull',
                           'gennorm'])

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

plt.savefig('Densidad.jpeg', format='jpeg', dpi=1300)


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

#Corte para valor extremo

logeados2 = logeados[ logeados >= np.quantile(logeados, 0.95)]

f2 = Fitter(logeados2,  
           distributions= ['gamma', 'dweibull',
                           'gennorm'])
f2.fit()

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










#%%

mes = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
       "julio", "agosto", "setiembre", "octubre", "noviembre", "diciembre"]

fechas = np.array([datetime.datetime.strptime(str(datos.FechaRegistro[0]), 
                               "%Y-%m-%d %H:%M:%S") ])
for i in range(0, len(datos.FechaRegistro)):
    temp = datetime.datetime.strptime(str(datos.FechaRegistro[i]), 
                               "%Y-%m-%d %H:%M:%S")
    fechas = np.append(fechas, temp)

meses = [mes[x.month] for x in fechas]


#%%

sns.countplot(meses)
plt.title("Histograma de meses de registro")
plt.xlabel('Mes')
plt.ylabel('Conteo') 

plt.savefig('Conteo_meses_registro.jpeg', format='jpeg', dpi=1300)


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
    



