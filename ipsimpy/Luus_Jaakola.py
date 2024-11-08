# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:38:05 2020

@author: BrunoLugao
"""

from numpy import empty, random, dot, diag, copy, savetxt, array, append
from os import mkdir, path, chdir
from time import time
from sys import stdout


class opcoes:
    Usuario={"Nout":50,
            "Nint":100,
            "Epsilon":0.05,
            "Tolerancia":10**-1,
            "VerOpcoes":False}

    @classmethod
    def Definicao_Parametros_Controle(self,Dicionario):
        if Dicionario==None:
            return

        if type(Dicionario)!=dict:
            raise Exception("Erro: Os parametros de controle devem estar no formato de dicionario")

        for nome in Dicionario:
            if nome in self.Usuario:
                self.Usuario[nome]=Dicionario[nome]
            else:
                print("Erro: A Chave",nome,"Esta Escrita de Modo Errado ou Não Existe")
                print("O Padrão das Chaves é")
                print(self.Usuario.keys())
                return None
            
    @classmethod
    def Validacao_Parametros_Controle(self):
        if type(self.Usuario["Nout"])!=int or self.Usuario["Nout"]<=0:
            print("Erro: O valor de Nout deve ser um inteiro maior que zero!")
            return None

        if type(self.Usuario["Nint"])!=int or self.Usuario["Nint"]<0:
            print("Erro: O valor de Nint deve ser um inteiro maior que zero!")
            return None

        if type(self.Usuario["Epsilon"])!=float or self.Usuario["Epsilon"]<=0 or self.Usuario["Epsilon"]>=1:
            print("Erro: O valor de Epsilon deve ser um real ente 0 e 1!")
            return None

        if type(self.Usuario["Tolerancia"])!=float:
            print("Erro: O valor da Tolerância deve ser um real")
            return None

        if type(self.Usuario["VerOpcoes"])!=bool:
            print("Erro: VerOpcoes deve ser True or False")
            return None
        
        
class Luus_Jaakola(opcoes):
    Defaut={"Nout":50,
        "Nint":100,
        "Epsilon":0.05,
        "Tolerancia":10**-1,
        "VerOpcoes":False}
    
    Convergencia=array([])
    Parametros_Estimados=None
    Valor_FO=None
    Amplitude=None
    Custo_Computacional=None
    
    def __init__(self,Funcao_Objetivo,Intervalo_Busca,OpcoesLJ=None):
        self.Definicao_Parametros_Controle(self.Defaut)
        self.Definicao_Parametros_Controle(OpcoesLJ)
        self.Validacao_Parametros_Controle()
        
        Np=len(Intervalo_Busca)
        Amplitude=empty(Np)
        P0=empty(Np)
        convergencia=array([])
        
        ini=time()
        for i in range(Np):
            Amplitude[i]=Intervalo_Busca[i][1] - Intervalo_Busca[i][0]
            P0[i]=random.uniform(Intervalo_Busca[i][0],Intervalo_Busca[i][1])

            if Amplitude[i]<0:
                print("\033[1;31m Erro: Algum Elemento do Intervalo de Busca Esta Trocado! \033[0;0m")
                return None

        Pold=copy(P0)
        Pbest=copy(P0)
        Sbest=Funcao_Objetivo(Pbest)
        
        i=0
        while i<self.Usuario["Nout"] and Sbest>=self.Usuario["Tolerancia"]:
            for j in range(self.Usuario["Nint"]):
                Pnew=Pold + dot(diag(random.uniform(-0.5,0.5,Np)),Amplitude)
                Snew=Funcao_Objetivo(Pnew)
                if Snew < Sbest:
                    Pbest=Pnew
                    Sbest=Snew
                    Pold=Pnew

            Amplitude=(1-self.Usuario["Epsilon"])*Amplitude
            convergencia = append(convergencia, Sbest)
            stdout.write("\rNout: %d" % (i+1))
            stdout.flush()
            i=i+1
            
        fim=time()
        stdout.write("\n")
        
        self.Parametros_Estimados=copy(Pbest)
        self.Valor_FO=Sbest
        self.Convergencia=copy(convergencia)
        self.Amplitude=Amplitude
        self.Custo_Computacional=fim-ini
        
        if self.Usuario["VerOpcoes"] == True:
            print("")
            print("OPÇÕES Luus-Jaakola")
            print("Nout={}".format(self.Usuario["Nout"]))
            print("Nout Utilizado={}".format(i))
            print("Nint={}".format(self.Usuario["Nint"]))
            print("Fator de Contracao={}".format(self.Usuario["Epsilon"]))
            print("Amplitide do Intervalo de Busca={}".format(self.Amplitude))
            print("Valor Estimado={}".format(Pbest))
            print("Valor na FO={0:.6e}".format(Sbest))
            print("Tolerância={0:.0e}".format(self.Usuario["Tolerancia"]))
            print("")
            
        Diretorio_Resultados="Arquivos Saida LJ"

        if path.isdir(Diretorio_Resultados):
            chdir(path.abspath(Diretorio_Resultados))
        else:
            mkdir(Diretorio_Resultados)
            chdir(path.abspath(Diretorio_Resultados))
            

        arquivo=open("DetalhesLJ.txt","w")
        arquivo.write("")
        arquivo.write("OPÇÕES Luus-Jaakola\n")
        arquivo.write("Nout={}\n".format(self.Usuario["Nout"]))
        arquivo.write("Nout Utilizado={}\n".format(i))
        arquivo.write("Nint={}\n".format(self.Usuario["Nint"]))
        arquivo.write("Fator de Contracao={}\n".format(self.Usuario["Epsilon"]))
        arquivo.write("Amplitide do Intervalo de Busca={}\n".format(self.Amplitude))
        arquivo.write("Valor Estimado={}\n".format(Pbest))
        arquivo.write("Valor na FO={0:.6e}\n".format(Sbest))
        arquivo.write("Tolerância={0:.0e}\n".format(self.Usuario["Tolerancia"]))
        arquivo.write("")
        arquivo.close()

        savetxt("Parametros_EstimadosLJ.txt",Pbest,delimiter='\t')
        savetxt("Valor FO LJ.txt",array([Sbest]))
        savetxt("ConvergenciaLJ.txt",convergencia,delimiter='\t')

        chdir(path.abspath(".."))
      