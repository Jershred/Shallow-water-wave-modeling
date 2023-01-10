# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 09:58:54 2022
@author: Jérémy Archier - jeremy.archier@etu.univ-lyon1.fr

Résolution de l'équation de Korteweg–De Vries 2D par méthode spectrale avec la 
formulation Fourier-Galerkin et le schéma d'intégration AB3.

Paramètres:
    
    - N : le nombre de mode
    - Nt : le nombre d'itérations temporelle
    - x : le domaine de calcul
    - k : le nombre d'onde
    - u0 : la condition de vitesse initiale
    - uk : les coefficients de fourrier initiaux
    - u : les vitesses moyennes du fluide
    - dt : le pas de discrétisation en temps

All rights reserved
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d  # Fonction pour la 3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#--------------------------Définition des fonctions----------------------------

def WK(uk,k,m):
    """Fonction calculant la transformé de Fourier du terme non linéaire u^2/2
    que l'on nomme w"""
    
    u = np.fft.ifft(uk) #coefficient de fourrier initiaux
    w = 0.5*u*u
    wk = np.fft.fft(w)
    # Kmax = 2.0/3.0*max(k) #échelle de coupure de desaliasage selon Orzag
    # for i in range (len(k)):
    #     if (abs(k[i])>Kmax): #boucle sur tous le modes
    #             wk[i]=0.0 #j'annule tous les modes aliasé
                
    return wk

def H(uk,k,m):
    """Fonction calculant l'opérateur linéaire et non linéaire pour un sclaire uk"""
    
    return 1j*(-k*WK(uk,k,m) + k**3*uk - m**2/k*uk)

def verifstab(u,k,m,dt):
    """Fonction traçant les valeurs propres afin de voir si elles sont bien 
    dans le domaine de stabilité du schéma que l'on souhaite utiliser."""
    
    Um = np.amax(abs(u)) #majoration de u(x)
    Lh = np.diag(1j*(-k*Um+k**3-m**2/j))
    Lambda = np.linalg.eigvals(dt*Lh)
    plt.plot(Lambda.real,Lambda.imag, '+b', label="eigenvalues")
    plt.xlabel("Real axis",fontsize=18)
    plt.ylabel("Imaginary axis",fontsize=18)
    plt.title("Spectrum")
    plt.legend()
    plt.grid()
    plt.show()

def resolution(x,Nt,N,dt,u0,uk,k,m):
    """Fonction résolvant l'équation de Korteweg–De Vries selon la 
    formaultation Fourier-Galerkin AB3 pour un nombre d'itérations Nt donnés
    avec un pas dt pour une solution initiale u0"""
    
    #Schéma d'Euler explicite pour les 3 premiers termes
    for j in range (Nt-1):
        for i in range(N):
            uk[i,:,j+1] = uk[i,:,j]+dt*H(uk[i,:,j],k,m)
        
        u = np.zeros((N,N))
            
        for i in range(N):
            u[i] = np.fft.fft(uk[i,:,j+1]) #coefficient de fourrier initiaux
            
            
        figure(j)
        gca(projection='3d').plot_surface(X,Y,u,cmap=cm.coolwarm, linewidth=0)
        xlabel('X(m)')
        ylabel('Y(m)')
        

    #Schéma AB3
    for j in range (3,Nt-1):
        for i in range(N):
            #calcul de u_k^(n+1)
            uk[i,:,j+1] = uk[i,:,j] + dt/12.0*(23.0*H(uk[i,:,j],k,m)-16.0*H(uk[i,:,j-1],k,m)+5.0*H(uk[i,:,j-2],k,m))
            
        #reconstruction de la solution toutes les 1ms
        #if (i*dt%0.001==0):
        for i in range(N):
            u[i] = np.fft.fft(uk[i,:,j+1]) #coefficient de fourrier initiaux

        figure(j)
        gca(projection='3d').plot_surface(X,Y,u,cmap=cm.coolwarm, linewidth=0)
        xlabel('X(m)')
        ylabel('Y(m)')
        
            
    return uk

###############################################################################
#############################Programme principal###############################
###############################################################################

#---------------------------Points de collocation------------------------------

N = 16 #le nombre de mode
x = np.linspace(-np.pi,np.pi, N,endpoint=False) #domaine de calcul
y = np.linspace(-np.pi,np.pi, N,endpoint=False) #domaine de calcul
k = np.fft.fftfreq(N)*N
m = np.fft.fftfreq(N)*N

X, Y = np.meshgrid(x, y)

Nt = 10; #nombre d'iteration temporelle
dt = 0.00001

sigma, mu = 1.0, 0.0

#----------------------------Conditions initiale-------------------------------

u0 = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        u0[i,j] = np.exp(-( (np.sqrt(x[i]*x[i]+y[j]*y[j])-mu)**2 / ( 2.0 * sigma**2 ) ) )
        

X, Y = np.meshgrid(x, y)


figure(0)

gca(projection='3d').plot_surface(X,Y,u0,cmap=cm.coolwarm, linewidth=0)
xlabel('X(m)')
ylabel('Y(m)')

#-----------------------------Tableaux initiaux--------------------------------

uk = np.zeros((N,N,Nt), dtype=complex)

for i in range(N):
        uk[i,:,0] = np.fft.fft(u0[i]) #coefficient de fourrier initiaux

#------------------------Vérification de la stabilité--------------------------

#verifstab(u0,k,m,dt)

#---------------------------------Résolution-----------------------------------

resolution(x,Nt,N,dt,u0,uk,k,m)

