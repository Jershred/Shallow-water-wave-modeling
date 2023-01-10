# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 18:02:25 2022

@author: Jérémy Archier - jeremy.archier@etu.univ-lyon1.fr

Résolution de l'équation de Korteweg–De Vries 1D par méthode spectrale avec la 
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

def verifstab(u,k,dt):
    """Fonction traçant les valeurs propres afin de voir si elles sont bien 
    dans le domaine de stabilité du schéma que l'on souhaite utiliser."""
    
    Um = max(abs(u)) #majoration de u(x)
    Lh = np.diag(1j*k*(k*k-6*Um))
    Lambda = np.linalg.eigvals(dt*Lh)
    plt.plot(Lambda.real,Lambda.imag, '+b', label="eigenvalues")
    plt.xlabel("Real axis",fontsize=18)
    plt.ylabel("Imaginary axis",fontsize=18)
    plt.title("Spectrum")
    plt.xlim([-1.5, 1.5])
    plt.ylim([0.0, 1.5])
    plt.legend()
    plt.grid()
    plt.show()

def resolution(x,Nt,N,dt,u0,uk,k):
    """Fonction résolvant l'équation de Korteweg–De Vries selon la 
    formaultation Fourier-Galerkin AB3 pour un nombre d'itérations Nt donnés
    avec un pas dt pour une solution initiale u0"""
    
    eps = 1.0e-10 #Epsilon pour reconstituer la solution toutes les x ms
    
    #Schéma d'Euler explicite pour les 3 premiers termes
    for j in range (Nt-1):
        uk[:,j+1] = uk[:,j]+dt*H(uk[:,j],k)

    #Schéma AB3
    for i in range (3,Nt-1):
            #calcul de u_k^(n+1)
            uk[:,i+1] = uk[:,i] + dt/12.0*(23.0*H(uk[:,i],k)-16.0*H(uk[:,i-1],k)+5.0*H(uk[:,i-2],k))
            
            #reconstruction de la solution toutes les 10ms
            if (abs(i*dt%0.02)<eps): #pas d'égalité sur les réels
                u=np.fft.ifft(uk[:,i])
                plt.plot(x,u0,'-b',x,u.real,'-r')
                plt.title(r'$t={:.2f}$ s'.format((i+1)*dt))
                plt.savefig('film/kdv{}.png'.format(i)) #Sauvegarde les graphs dans un dossier
                plt.show()
            
    return uk


###############################################################################
#############################Programme principal###############################
###############################################################################

#---------------------------Points de collocation------------------------------

N = 64 #le nombre de mode
x = np.linspace(-np.pi,np.pi, N,endpoint=False) #domaine de calcul
k = np.fft.fftfreq(N)*N

Nt = 200000 #nombre d'iteration temporelle
dt = 0.00001 #pas de temps maximale pour convergence

#----------------------------Conditions initiale-------------------------------

#u0 = np.sin(x) #condition initiale cosinus
#u0 = 4.0*1.0/(np.cosh(x))**2 #condition pour solution exacte
u0 = 9.0/2.0/(np.cosh(3.0/2.0*x))**2 #condition pour solution exacte
#u0 = 4.0*1.0/(np.cosh(np.sqrt(2)*x))**2 + 2.0/(np.cosh(1.0/np.sqrt(2)*x))**2 #condition pour solution exacte

#-----------------------------Tableaux initiaux--------------------------------

uk = np.zeros((N,Nt), dtype=complex)

uk[:,0] = np.fft.fft(u0) #coefficient de fourrier initiaux

#------------------------Vérification de la stabilité--------------------------

verifstab(u0,k,dt)

#---------------------------------Résolution-----------------------------------

resolution(x,Nt,N,dt,u0,uk,k)