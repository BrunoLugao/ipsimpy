# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:59:22 2019

@author: Bruno
"""

from ipsimpy.Problema_Inverso import Definicoes_Preliminares as _Df
from numpy import array, eye, linalg, dot, zeros, append, copy, savetxt
from os import mkdir, path, chdir
from time import time
from sys import stdout


class opcoes:
    Usuario={"Omega":10,
             "MaxIter":10,
             "Tolerancia":10**-5,
             "VerOpcoes":False}

    @classmethod
    def Definicao_Parametros_Controle(self,Dicionario):
        if Dicionario==None:
            return

        if type(Dicionario)!=dict:
            raise Exception("Os parametros de controle devem estar no formato de dicionario")

        for nome in Dicionario:
            if nome in self.Usuario:
                self.Usuario[nome]=Dicionario[nome]
            else:
                print("A Chave",nome,"Esta Escrita de Modo Errado ou Não Existe")
                print("O Padrão das Chaves é")
                print(self.Usuario.keys())
                raise Exception

    @classmethod
    def Validacao_Parametros_Controle(self):
        if self.Usuario["Omega"]<=0:
            raise Exception("O valor de Omega deve ser maior que Zero")

        if type(self.Usuario["MaxIter"])!=int or self.Usuario["MaxIter"]<=0:
            raise Exception("O valor de MaxIter deve ser um inteiro maior que Zero")

        if type(self.Usuario["Tolerancia"])!=float:
            raise Exception("O valor de w deve ser um real")

        if type(self.Usuario["VerOpcoes"])!=bool:
            raise Exception("VerOpcoes deve ser True or False")

class metodos(opcoes):

    @classmethod
    def Jacobiana(self,Parametros,Delta_h):
        h=Delta_h
        Nt=len(_Df.Tempos_Exp)
        Np=len(Parametros)
        jac=zeros([Np,Nt])
        incremento=h*eye(Np)
        Z=Parametros
        for i in range(Np):
            Z1=Z+incremento[i,:]
            Z2=Z-incremento[i,:]
            f1=_Df.Solucao_PD(Z1)
            f2=_Df.Solucao_PD(Z2)
            jac[i,:]=(f1-f2)/(2*h[i])

        return jac.T


class Levemberg_Marquardt(metodos):
    Defaut={"Omega":10,
            "MaxIter":10,
            "Tolerancia":10**-5,
            "VerOpcoes":False}

    Convergencia=array([])
    Parametros_Estimados=None
    Valor_FO=None
    Custo_Computacional=None

    def __init__(self,FuncaoObjetivo,ChuteInicial,Passo,OpcoesLM=None):
        self.Definicao_Parametros_Controle(self.Defaut)
        self.Definicao_Parametros_Controle(OpcoesLM)
        self.Validacao_Parametros_Controle()

        omega=self.Usuario["Omega"]
        convergencia=array([])
        aux=10
        P0=ChuteInicial
        Ik=eye(len(P0))
        Pk=P0
        Sk=FuncaoObjetivo(Pk)
        k=0

        ini=time()
        while k<self.Usuario["MaxIter"] and Sk>=self.Usuario["Tolerancia"]:
            D_Calc=_Df.Solucao_PD(Pk)
            Sk=FuncaoObjetivo(Pk)
            J=self.Jacobiana(Pk,Passo)
            A=dot(J.T,J)+omega*Ik
            b=-(dot(J.T,D_Calc-_Df.Dados_Exp))
            DeltaP=linalg.solve(A, b)
            Pnovo=Pk+DeltaP
            Snovo=FuncaoObjetivo(Pnovo)
            if Snovo>Sk:
                omega=omega*aux
            else:
                omega=omega/aux
                Pk=Pnovo
                Sk=Snovo
            convergencia=append(convergencia,Snovo)
            stdout.write("\rIteração: %d" % (k+1))
            stdout.flush()
            k=k+1
            
        fim=time()
        stdout.write("\n")
        
        self.Parametros_Estimados=Pk
        self.Valor_FO=Sk
        self.Convergencia = copy(convergencia)
        self.Custo_Computacional=fim-ini

        if self.Usuario["VerOpcoes"]==True:
            print(" ")
            print("OPÇÕES LEVEMBERG MARQUARDT")
            print("omega inicial={0:.0e}".format(self.Usuario["Omega"]))
            print("omega final={0:.0e}".format(omega))
            print("Pk=",Pk)
            print("Tolerância={0:.0e}".format(self.Usuario["Tolerancia"]))
            print("Valor na FO={0:.5e}".format(Sk))
            print("Numero Máximo de Iterações=",self.Usuario["MaxIter"])
            print("Numero de Iterações Utilizado=",k)
            print("Passo h={}".format(Passo))
            print(" ")

        Diretorio_Resultados="Arquivos Saida LM"

        if path.isdir(Diretorio_Resultados):
            chdir(path.abspath(Diretorio_Resultados))
        else:
            mkdir(Diretorio_Resultados)
            chdir(path.abspath(Diretorio_Resultados))

        arquivo=open("DetalhesLM.txt","w")
        arquivo.write("OPÇÕES LEVEMBERG MARQUARDT\n")
        arquivo.write("omega inicial={0:.0e}\n".format(self.Usuario["Omega"]))
        arquivo.write("omega final={0:.0e}\n".format(omega))
        arquivo.write("Pk={}\n".format(Pk))
        arquivo.write("Tolerância={0:.0e}\n".format(self.Usuario["Tolerancia"]))
        arquivo.write("Valor na FO={0:.5e}\n".format(Sk))
        arquivo.write("Numero Máximo de Iterações={}\n".format(self.Usuario["MaxIter"]))
        arquivo.write("Numero de Iterações Utilizado={}\n".format(k))
        arquivo.write("Passo h={}\n".format(Passo))
        arquivo.close()

        savetxt("Parametros_EstimadosLM.txt",Pk,delimiter='\t')
        savetxt("Valor FO LM.txt",array([Sk]))
        savetxt("ConvergenciaLM.txt",convergencia,delimiter='\t')

        chdir(path.abspath(".."))
