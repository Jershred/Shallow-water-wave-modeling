# Shallow-water-wave-modeling
## Context
Project carried out for the course of Numerical methods for Fluid dynamics in Python at Université Lyon 1 (France).

| Made by | Referent teacher | 
| ------------- |:-------------:|
| Jérémy Archier | Mr. Bastien Di Pierro |

[Subject](Report/Combustion_BE_Numerics_ressource_AG.pdf)

[Report](Report/Report_CombustionForPropulsion_PracticalSession1_Archier_Bourakkadi.pdf)


## Project
### Introduction
The Fourier Spectral method is a method that can be used to solve partial differential equations, such as the 1D and 2D Korteweg–De Vries equation, which is a mathematical model that describes the behavior of shallow water waves. This method utilizes the properties of the Fourier transform to represent functions in the spatial domain as a series of complex exponentials in the frequency domain.

By applying the Fourier Spectral method, one can numerically approximate the solution of the Korteweg–De Vries equation. Additionally, the solution obtained by the Fourier Spectral method is then integrated over time using a specific method called "AB3" which stands for "Adams-Bashforth, 3rd order" numerical integration.

The implementation of the Fourier Spectral method and the time integration AB3 can be done using the Python programming language
