# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:45:57 2020

@author: BrunoLugao
"""
from ipsimpy.Problema_Inverso import Definicoes_Preliminares
from numpy import array, float64, zeros, eye

class Analise_Sensibilidade:

    Matriz=array([], dtype=float64)
    Parametro=None
    
    def __init__(self,Solucao_PD,Parametros,Tempos_Sensibilidade,Delta_h):
        h=Delta_h
        Nt_Sensibilidade=len(Tempos_Sensibilidade)
        Np=Definicoes_Preliminares.Np
        jac=zeros([Np,Nt_Sensibilidade])
        incremento=h*eye(Np)
        Z=Parametros
        
        for i in range(Np):
            Z1=Z+incremento[i,:]
            Z2=Z-incremento[i,:]
            f1=Solucao_PD(Z1)
            f2=Solucao_PD(Z2)
            jac[i,:]=(f1-f2)/(2*h[i])

        self.Matriz = jac.T
        
        self.Parametro=tuple(jac)

        