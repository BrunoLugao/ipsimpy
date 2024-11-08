# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:42:37 2020

@author: BrunoLugao
"""
from time import time
from sys import stdout
from ipsimpy.Problema_Inverso import Definicoes_Preliminares
from numpy import array, float64, append, random, log, dot, diag, linalg, eye, mean, std, delete, zeros, s_, floor, sort


class MCMC_Sem_Priori:
    
    Cadeias=array([],dtype=float64)
    Taxa_Aceitação=0.0
    Custo_Computacinal=0.0
    Media_Parametro=array([],dtype=float64)
    Desvio_Parametro=array([],dtype=float64)
    Intervalo_Confianca=None
    
        
    def __init__(self,Solucao_PD,Estado_Inicial,Nmcmc,Nburnin,Passo):
        Np=Definicoes_Preliminares.Np
        aceito=0
        
        ze=Definicoes_Preliminares.Dados_Exp
        Vinv=Definicoes_Preliminares.Matriz_Cov_Dados_Exp
        Zold=array([Estado_Inicial])
        zmold=Solucao_PD(Zold.flatten())
        Zmcmc=Zold

        
        st=diag(Passo)
        t1=time()
        for i in range(Nmcmc):
            Zs=random.multivariate_normal(Zold.flatten(), st).reshape(-1,Zold.size)
            zms=Solucao_PD(Zs.flatten())
            earg=-0.5*(dot(dot(ze-zms,Vinv),ze-zms))+0.5*(dot(dot(ze-zmold,Vinv),ze-zmold))
            U=log(random.uniform(0,1))
            if earg>=U:
                Zmcmc=append(Zmcmc,Zs,axis=0)
                Zold=Zs
                zmold=zms
                aceito=aceito+1
            else:
                Zmcmc=append(Zmcmc,Zold,axis=0)
                
            stdout.write("\rEstado: %d" % (i+1))
            stdout.flush()

        t2=time()
        stdout.write("\n")

        self.Custo_Computacinal=t2-t1
        self.Taxa_Aceitação=aceito/Nmcmc
        self.Cadeias=Zmcmc

        self.Media_Parametro=zeros(Np)
        self.Desvio_Parametro=zeros(Np)

        aux_confianca=zeros((Np,2))

        q = 0.95
        p = (1-q)/2.0
        m = Nmcmc - Nburnin + 1
        n = p*m
        n = floor(n)

        for i in range(Np):
            aux=delete(Zmcmc[:,i],s_[0:Nburnin+1])
            self.Media_Parametro[i]=mean(aux)
            self.Desvio_Parametro[i]=std(aux)

            aux_sort=sort(aux)
            aux_confianca[i,0]=aux_sort[0]
            aux_confianca[i,1]=aux_sort[-1]

        self.Intervalo_Confianca=tuple(aux_confianca)