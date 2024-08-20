# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:29:14 2024

@author: sergiozc
"""

# CANCELADOR DE ECO ACUSTICO

'''
    SERGIO ZAPATA CAPARRÓS
    
'''
    
#importar modulos completos
import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def NMLS(remota,signal,p,u):
    
    '''Función que implementa el algoritmo NMLS con argumentos de entrada la señal
    remota, la señal deseada, el orden del filtro y el parámetro de convergencia.
    Devuelve el filtro de wiener óptimo y la señal error'''
    
    N = len(remota)
    w = np.zeros((N-p+1,p)) #Matriz de coeficientes
    x = np.zeros(p) #Buffer de muestras donde se guardan muestras de la señal
    e = np.zeros(N) #Error
    d = np.zeros(p) #Guardamos la señal signal, la señal con eco
    
    for k in range(N-p):
    
        x = np.roll(x,1)
        x[0] = remota[k]
    

        d = signal[k]
    
    
        e[k] = d - np.dot(w[k,:],x)
    
    
        w[k+1,:] = w[k,:] + ((2*u*e[k]*x)/(sum(abs(x**2)) + 1e-25))
    
    return w, e

def DTD(remota,signal,p,u):
    
    '''Función implementa el algoritmo de doble talk detector con los mismos 
    argumentos de entrada y de salida que el NMLS'''
    
    N = len(remota)
    w=np.zeros((N-p+1,p)) #Matriz de coeficientes
    x=np.zeros(p) #Buffer de muestras donde se guardan muestras de la señal
    e=np.zeros(N) #Error
    d=np.zeros(p) #Guardamos la señal signal, la señal con eco
    
    for k in range(N-p):
    
        x=np.roll(x,1)
        x[0]=remota[k]
    

        d=signal[k]
    
    
        e[k]=d-np.dot(w[k,:],x)
    
    
        if k<2150:
            w[k+1,:]=w[k,:]+((2*u*e[k]*x)/(sum(abs(x**2))+1e-25))
        if k>=2150:
            w[k+1,:]=w[2150,:]
            
    return w, e
    

def calcula_maximo(SNR):
    '''Función que calcula el orden del filtro óptimo y el parámetro de convergencia
    óptimo'''
    
    maximo=SNR.max()
    [u_optimo, p_optimo]=np.where(SNR==maximo)
    u_optimo=u_array[u_optimo]
    p_optimo=p_array[p_optimo]
    u_optimo = u_optimo[0]
    p_op = p_optimo[0]
    
    return p_op, u_optimo

#PARTE 1: SNR SIN CANCELADOR DE ECO

#Cargamos las muestras de las señales

Fs_remota, remota = wavfile.read('remota.wav')
remota = remota.astype(np.float)

Fs_local, local = wavfile.read('local.wav')
local = local.astype(np.float)

Fs_signal, signal = wavfile.read('signal.wav')
signal = signal.astype(np.float)

#Esta es la SNR de la que partimos, con eco
SNR = 10*np.log10(sum(local**2)/sum((local-signal)**2))
print("La SNR sin el cancelador de eco es:", SNR, 'dB')



#Vamos a llevar a cabo el algoritmo NLMS, inicializaremos el filtro
#como un vector de coefientes todos cero e iremos calculando los siguientes 
#coeficientes con el algoritmo. Calcularemos todos los filtro, para los 
#ordenes del filtro p y parametro de convergencia u, para finalmente obtener
#cuales son los parametros optimos que nos proporcione la mayor SNR.

#Inicializamos variables
N = len(remota)

#Parametro de convergencia
u_array = np.array([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256])

#Orden del filtro
p_array = np.array([2, 3, 4, 5, 6, 7, 8])

#SNR
SNR_canc = np.zeros((len(u_array), len(p_array))) #Inicializamos la SNR ya que será matricial

#Llevamos a cabo el algoritmo y calculamos las SNRs para encontrar la mejor
for i in range (len(u_array)):
    
    u = u_array[i] #Para cada parámetro de convergencia
    
    for j in range (len(p_array)):
        
        p = p_array[j] #Para cada orden del filtro
        
        w, e = NMLS(remota, signal, p, u)
        
        #Calculamos el valor de la SNR para cada p y cada u, de forma que 
        #podremos obtener el valor optimo.
        SNR_canc[i,j] = 10*np.log10(sum(local**2)/sum((local-e)**2))
        
 
#Una vez calculado la SNR para cada orden del filtro y cada parametro de 
#convergencia, vamos a obtener los valores optimos de estos para los cuales
#la SNR es maxima:

p_op, u_optimo = calcula_maximo(SNR_canc)

print('La SNR para p =', p_op,' y para u =', u_optimo,'sale ' + str(SNR_canc[0,6]), 'dB')


#Comprobamos que obtenemos una mejora de la SNR considerable a la SNR sin
#cancelador de eco

#Por ultimo, vamos a volver a calcular los coeficientes del filtro para
#esta vez solo para los valores de p y u optimos.

#Con estos valores óptimos, calculamos el filtro de wiener
w_op, e_op = NMLS(remota,signal,p_op,u_optimo)

file_name = 'NMLS'
wavfile.write('resultados/' + file_name + '.wav', 8000, e_op.astype('int16'))

#Representamos la evolucion de los coeficientes para la mejor cofiguracion
#de u=0.001 y p=8, que nos da la mejor SNR

plt.figure(1)
plt.xlabel('k')
plt.ylabel('Coeficiente')
plt.title('Coeficientes para los valores optimos de p=8 y u=0.001')
for i in range (p_op):
    plt.plot(w_op[:,i])

#Observamos cómo los coeficientes se van adaptando conforme el nivel de ruido de
#la señal, cuanto más ruido el filtro filtra más, por lo que los coeficientes son
#más restrictivos. Vemos cómo todos los coeficientes para distinto orden del filtro,
#evolucionan de la misma forma.


# PARTE 2: DOUBLE TALK DETECTOR (DTD)

#Realizamos el algoritmo de igual manera que en el caso anterior, con la 
#salvedad de que ahora el detector detecta la voz local a partir de la muestra
#2150, de forma que el filtro solo se adaptara hasta esa muestra y apartir de 
#ahi mantenemos el ultimo cancelador obtenido.


#Inicializamos variables

N = len(remota)
#Parametro de convergencia
u_array = np.array([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256])
 
#Orden del filtro
p_array = np.array([2, 3, 4, 5, 6, 7, 8])

SNR_DTD = np.zeros((len(u_array), len(p_array)))

#Llevamos a cabo el algoritmo y calculamos las SNRs para encontrar la mejor
for i in range (len(u_array)):

    u = u_array[i]
    
    
    for j in range (len(p_array)):
        
        p = p_array[j]
        
        w, e=DTD(remota, signal, p, u)
        
        #Calculamos el valor de la SNR para cada p y cada u, de forma que 
        #podremos obtener el valor optimo.
        SNR_DTD[i,j]=10*np.log10(sum(local**2)/sum((local-e)**2))
        #print('La SNR para p=', p,' y para u=', u,'sale ' + str(SNR_DTD[i,j]))


#Calculamos los valores optimos de p y u, teniendo en cuenta la posicion
#de la matriz de SNRs calculadas sabuendo que las columnas se corresponden
#con los ordenes del filtro y las filas con cada parametro de convergencia.
p_op,u_optimo = calcula_maximo(SNR_DTD)

print('La SNR en DTD para p =', p_op,' y para u =', u_optimo,'sale ' + str(SNR_DTD[3,3]), 'dB')
#Ahora vamos a volver a calcular los coeficientes del filtro pero
#esta vez solo para los valores de p y u optimos.

w_optimo, e_optimo = DTD(remota,signal,p_op,u_optimo)

file_name = 'DTD'
wavfile.write('resultados/' + file_name + '.wav', 8000, e_optimo.astype('int16'))


#Representamos la evolucion de los coeficientes para la mejor cofiguracion
#de u=0.008 y p=5, que nos da la mejor SNR
plt.figure(2)
plt.xlabel('k')
plt.ylabel('Coeficiente')
plt.title('Coeficientes para los valores optimos con DTD de p=5 y u=0.008')
for i in range (p_op):
    plt.plot(w_optimo[0:2150,i])
    
#Como observamos ahora el valor optimo tiene un orden del filtro mas pequeño y
#un parametro de convergencia mayor ya que podremos converger mucho mas rapido
#Lo apreciamos también en los coeficientes, ya que llegados a un determinado valor
#se mantienen aproximadamente constantes

#Vamos a obtener la respuesta en frecuencia del cancelaor FIR para los valores
#optimos de p y u
omega, h = scipy.signal.freqz(w_optimo[2150,:],[1], worN=1024)  
plt.figure(3)
plt.plot(omega, 20*np.log10(abs(h)))
plt.title('Respuesta en frecuencia para filtro óptimo')
plt.xlabel('Frecuencia (rad/s)')
plt.ylabel('Magnitud (dB)')
plt.grid()

fase = np.unwrap(np.angle(h))
plt.figure(4)
plt.plot(omega,fase)
plt.title('Fase filtro óptimo')
plt.xlabel('Frecuencia (rad/s)')
plt.ylabel('Fase (grados)')
plt.grid()

#Podemos observar que nos encontramos con un filtro paso baja que se adapta a la señal
#y que filtra más si se encuentra con frecuencias a SNRs más bajas