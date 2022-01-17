# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 14:38:49 2018

@author: Bruno
"""

from numpy import array, float64, eye, linalg

class Definicoes_Preliminares:
    Dados_Exp=array([], dtype=float64)
    Tempos_Exp=array([], dtype=float64)
    Parametros=array([], dtype=float64)
    Np=0
    Pos_Exp=None
    Solucao_PD=None
    FO=None
    Matriz_Cov_Dados_Exp=array([], dtype=float64)
    
    
    def Set_Solucao_PD(funcao):
        """
            A SOLUÇÃO DO PD DEVE RETORNAR UM VETOR np.array DE DIMENSÃO (n,)
        """
        Definicoes_Preliminares.Solucao_PD=funcao
    
    def Set_Parametros(par):
        Definicoes_Preliminares.Parametros=par
    
    def Set_Posicao_Exp(pos):
        Definicoes_Preliminares.Pos_Exp=pos
        
    def Set_Dados_Exp(dados):
        Definicoes_Preliminares.Dados_Exp=array(dados)
        
    def Set_Tempos_Exp(tempos):
        Definicoes_Preliminares.Tempos_Exp=array(tempos)
    
    def Set_FuncaoObjetivo(funcao):
        Definicoes_Preliminares.FO=funcao

    def Set_Matriz_Covariancia(dados_exp,sigma_exp):
        Definicoes_Preliminares.Matriz_Cov_Dados_Exp=linalg.inv((sigma_exp**2)*eye(len(dados_exp)))

    def Set_Numero_Parametros(Np):
        Definicoes_Preliminares.Np=Np