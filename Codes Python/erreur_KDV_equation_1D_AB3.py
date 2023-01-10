# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:09:22 2022

@author: Jérémy Archier - jeremy.archier@etu.univ-lyon1.fr

Calcul de l'erreur temporelle de l'équation de Korteweg–De Vries 1D par méthode
spectrale avec la formulation Fourier-Galerkin et le schéma d'intégration AB3.

Pour simuler une solution exacte qui respecte les conditions aux limites 
périodiques, je fais un train de 2 ondes déphasées de 2 pi ce qui permet de 
répliquer la vraie solution pour 1-2 secondes de simulation. 

Je ne trouve pas l'ordre 3 attendu malgré les modifications effectués dans mon
code, l'erreur commise reste constante.

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

#--------------------------Définition des fonctions----------------------------

def WK(uk,k):
    """Fonction calculant la transformé de Fourier du terme non linéaire u^2/2
    que l'on nomme w"""
    
    u = np.fft.ifft(uk)
    w = 0.5*u*u
    wk = np.fft.fft(w)
    Kmax = 2.0/3.0*max(k) #échelle de coupure de desaliasage selon Orzag
    for i in range (len(k)):
        if (abs(k[i])>Kmax): #boucle sur tous le modes
            wk[i]=0.0 #j'annule tous les modes aliasé
            
    return wk

def H(uk,k):
    """Fonction calculant l'opérateur linéaire et non linéaire pour un sclaire uk"""
    
    return 1j*k**3*uk - 6j*k*WK(uk,k)

def resolution(x,Nt,N,dt,u0,uk,k):
    """Fonction résolvant l'équation de Korteweg–De Vries selon la 
    formaultation Fourier-Galerkin AB3 pour un nombre d'itérations Nt donnés
    avec un pas dt pour une solution initiale u0"""
    
    #Schéma d'Euler explicite pour les 3 premiers termes
    for j in range (Nt-1):
        uk[:,j+1] = uk[:,j]+dt*H(uk[:,j],k)

    #Schéma AB3
    for i in range (3,Nt-1):
            #calcul de u_k^(n+1)
            uk[:,i+1] = uk[:,i] + dt/12.0*(23.0*H(uk[:,i],k)-16.0*H(uk[:,i-1],k)+5.0*H(uk[:,i-2],k))
    
    u_out = np.fft.ifft(uk[:,Nt-1]).real
    
    return u_out

def sol_exacte(x,Nt,N,dt,u_ex):
    """Fonction construisant la solution exacte de KdV pour une condition 
    initiale u(x,0) = 2 sech^2(x)"""
    
    for t in range (Nt):
        #On simule les conditions aux limites périodique avec 2 trains d'onde déphasé de 2 pi
        u_ex[:,t] = 9.0/2.0*(np.cosh(3.0/2.0(x[:]-9.0*t*dt))**2 + 9.0/2.0*(np.cosh(3.0/2.0*(x[:]+2*np.pi-9.0*t*dt))**2
        
    u_ex_out = u_ex[:,Nt-1]    
        
    return u_ex_out

###############################################################################
#############################Programme principal###############################
###############################################################################

#---------------------------Points de collocation------------------------------

N = 64 #le nombre de mode
x = np.linspace(-np.pi,np.pi, N,endpoint=False) #domaine de calcul
k = np.fft.fftfreq(N)*N

#----------------------------Conditions initiale-------------------------------

u0 = 9.0/2.0/(np.cosh(3.0/2.0*x))**2 #condition pour solution exacte
    
#----------------------Tracer de l'ordre de la méthode-------------------------

DT = np.array([1.0e-7, 1.0e-6, 1.0e-5]); #tableau de différent pas de temps
err = np.zeros(len(DT)) #Erreur
t_sim = 0.001 #temps simulé

for i,dt in enumerate(DT): 
    
    Nt = int(t_sim/dt)
    t = np.linspace(0,Nt*dt, Nt,endpoint=False)
    
    u_ex = np.zeros((N,Nt), dtype=float) #solution exacte
    u = np.zeros((N,Nt), dtype=float) #solution numérique
    uk = np.zeros((N,Nt), dtype=complex)
    
    uk[:,0] = np.fft.fft(u0) #coefficient de fourrier initiaux
    
    u_ex_out = sol_exacte(x,Nt,N,dt,u_ex) #solution exacte à l'instant finale
    u_out = resolution(x,Nt,N,dt,u0,uk,k) #solution numérique à l'instant finale
    
    err[i] = np.linalg.norm(u_out-u_ex_out)/np.linalg.norm(u_ex_out); #erreur relative

#Calcul de la pente pour tous les points moyenné

pentes = np.empty(len(DT)-1) #pente de la droite de l'erreur

for i in range(len(err)-1):
    pentes[i] = (np.log(err[i+1])-np.log(err[i]))/(np.log(DT[i+1])-np.log(DT[i])) 

pente = np.average(pentes[:]) #pente moyenne de la courbe

#Tracé en échelle loglog de l'erreur

fig,ax = plt.subplots()

ax.loglog(DT, err, '-b', DT, DT**2,'-r', DT, DT**3, '-g')
ax.legend(('AB3','ordre2','ordre3'))
ax.set_xlabel(r'$\Delta t$',fontsize=18)
ax.set_ylabel(r'$\epsilon_{rel}$',fontsize=18)
ax.grid(which='both')
plt.show()