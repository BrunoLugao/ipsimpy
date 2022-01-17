# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 12:36:59 2018

@author: Bruno
"""

from numpy import array, zeros, random, append,copy, savetxt, sqrt
from os import mkdir, path, chdir
from time import time
from sys import stdout


class options:
    Usuario={"w":0.5,
            "c1":1.0,
            "c2":2.0,
            "MaxIter":100,
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
        if type(self.Usuario["w"])!=float or self.Usuario["w"]<=0 or self.Usuario["w"]>1.0:
            raise Exception("O valor de w deve ser um real entre 0.0 e 1.0")

        if type(self.Usuario["c1"])!=float or self.Usuario["c1"]<=0:
            raise Exception("O valor de c1 deve ser um real maior que Zero")

        if type(self.Usuario["c2"])!=float or self.Usuario["c2"]<=0:
            raise Exception("O valor de c2 deve ser um real maior que Zero")

        if type(self.Usuario["MaxIter"])!=int or self.Usuario["MaxIter"]<=0:
            raise Exception("O valor de MaxIter deve ser um inteiro maior que Zero")

        if type(self.Usuario["Tolerancia"])!=float:
            raise Exception("O valor de w deve ser um real")

        if type(self.Usuario["VerOpcoes"])!=bool:
            raise Exception("VerOpcoes deve ser True or False")

class Particulas(options):
    def __init__(self,x0):
        self.posicao_i=array([])
        self.velocidade_i=array([])
        self.melhor_posicao_i=array([])
        self.melhor_fitness_posicao_i=-1.0
        self.fitness_i=-1.0
        self.dimensao=len(x0)

        self.velocidade_i=random.uniform(-1,1,self.dimensao)
        self.posicao_i=copy(x0)

        #print(self.velocidade_i)
        #print(self.posicao_i)
        #print(aux_pos)
        #print(type(aux_pos))


    def Avaliar_Fitness(self,FO):
        self.fitness_i=FO(self.posicao_i)

        if self.fitness_i<self.melhor_fitness_posicao_i or self.melhor_fitness_posicao_i==-1.0:
            self.melhor_posicao_i=copy(self.posicao_i)
            self.melhor_fitness_posicao_i=self.fitness_i

    def Atualiza_Velocidade(self,Gbest):
        #wmax=self.Usuario["w"]
        #wmin=0.4
        w=self.Usuario["w"]
        c1=self.Usuario["c1"]
        c2=self.Usuario["c2"]
        
        #w = wmax - ((wmax-wmin)*Iter)/self.Usuario["MaxIter"]
        
        phi1=c1#Coloquei isso
        phi2=c2#Coloquei isso

        phi=phi1+phi2#Coloquei isso

        k=1.0#Coloquei isso

        chi=(2.0*k)/(abs(2.0-phi-sqrt(phi**2-4.0*phi)))#Coloquei isso

        for i in range(self.dimensao):
            r1=random.random()
            r2=random.random()
            vel_cognitiva=c1*r1*(self.melhor_posicao_i[i]-self.posicao_i[i])
            vel_social=c2*r2*(Gbest[i]-self.posicao_i[i])
            
            #if self.velocidade_i[i]>2:
            #    self.velocidade_i[i]=2
                
            #if self.velocidade_i[i]<-2:
            #    self.velocidade_i[i]=-2
            
            #No Fator de Constrição não aprece esse w, mas quando eu coloco ele melhora bastante
            self.velocidade_i[i]= chi*(w*self.velocidade_i[i]+vel_cognitiva+vel_social)

    def Atualiza_Posicao(self,limites):
        for i in range(self.dimensao):
            self.posicao_i[i]=self.posicao_i[i]+self.velocidade_i[i]

            if self.posicao_i[i]>limites[i][1]:
                self.posicao_i[i]=limites[i][1]

            if self.posicao_i[i]<limites[i][0]:
                self.posicao_i[i]=limites[i][0]

class Populacao_Inicial:
    def __new__(cls,tamanho_pop,limit):
        cls.pop=array([])
        dim=len(limit)
        Populacao_Ini=zeros([tamanho_pop,dim])
        for i in range(tamanho_pop):
            for j in range(dim):
                Populacao_Ini[i][j]=random.uniform(limit[j][0],limit[j][1])

        for i in range(tamanho_pop):
            cls.pop=append(cls.pop,Particulas(Populacao_Ini[i]))

        return cls.pop


class PSO(Particulas):
    #FAZENDO ASSIM SEMPRE QUE EU RODO O PSO ELE VOLTA PARA AS OPÇÕES PADRAO
    Defaut={"w":0.5,
            "c1":1.0,
            "c2":2.0,
            "MaxIter":100,
            "Tolerancia":10**-5,
            "VerOpcoes":False}
    Convergencia=array([])
    Parametros_Estimados=None
    Valor_FO=None
    Custo_Computacional=None

    def __init__(self,FuncaoObjetivo,numero_particulas,limites,OpcoesPSO=None):
        self.Definicao_Parametros_Controle(self.Defaut)
        self.Definicao_Parametros_Controle(OpcoesPSO)
        self.Validacao_Parametros_Controle()

        fitness_Gbest=-1.0
        posicao_Gbest=array([])

        convergencia=array([])

        enxame=Populacao_Inicial(numero_particulas,limites)

        enxame[0].Avaliar_Fitness(FuncaoObjetivo)
        posicao_Gbest=copy(enxame[0].posicao_i)
        fitness_Gbest=enxame[0].fitness_i

        i=0
        ini=time()
        while i<self.Usuario["MaxIter"] and fitness_Gbest>self.Usuario["Tolerancia"]:
            for j in range(numero_particulas):
                enxame[j].Avaliar_Fitness(FuncaoObjetivo)

                if enxame[j].fitness_i<fitness_Gbest:
                    posicao_Gbest=copy(enxame[j].posicao_i)
                    fitness_Gbest=enxame[j].fitness_i

            convergencia=append(convergencia,fitness_Gbest)
            
            for j in range(numero_particulas):
                enxame[j].Atualiza_Velocidade(posicao_Gbest)
                enxame[j].Atualiza_Posicao(limites)
            
            #print("Geracao=",i)
            stdout.write("\rGeração: %d" % (i+1))
            stdout.flush()
            i=i+1
            
        fim=time()
        #stdout.write("\n")
        
        self.Convergencia=copy(convergencia)
        self.Parametros_Estimados=posicao_Gbest
        self.Valor_FO=fitness_Gbest
        self.Custo_Computacional=fim-ini

        if self.Usuario["VerOpcoes"]==True:
            print("")
            print("OPÇÕES ENXAME DE PARTÍCULAS (PSO)")
            print("Numero de Particulas={}".format(numero_particulas))
            print("Numero Máximo de Iterações={}".format(self.Usuario["MaxIter"]))
            print("Numero de Iterações Utilizado={}".format(i))
            print("w={}".format(self.Usuario["w"]))
            print("c1={}".format(self.Usuario["c1"]))
            print("c2={}".format(self.Usuario["c2"]))
            print("Tolerancia={0:.0e}".format(self.Usuario["Tolerancia"]))
            print("Melhor Elemento do Enxame=",posicao_Gbest)
            print("Valor na FO={0:.6e}".format(fitness_Gbest))
            print("")

        Diretorio_Resultados="Arquivos Saida PSO"

        if path.isdir(Diretorio_Resultados):
            chdir(path.abspath(Diretorio_Resultados))
        else:
            mkdir(Diretorio_Resultados)
            chdir(path.abspath(Diretorio_Resultados))

        arquivo=open("DetalhesPSO.txt","w")
        arquivo.write("OPÇÕES ENXAME DE PARTÍCULAS (PSO)\n")
        arquivo.write("Numero de Particulas={}\n".format(numero_particulas))
        arquivo.write("Numero Máximo de Iterações={}\n".format(self.Usuario["MaxIter"]))
        arquivo.write("Numero de Iterações Utilizado={}\n".format(i))
        arquivo.write("w={}\n".format(self.Usuario["w"]))
        arquivo.write("c1={}\n".format(self.Usuario["c1"]))
        arquivo.write("c2={}\n".format(self.Usuario["c2"]))
        arquivo.write("Tolerancia={0:.0e}\n".format(self.Usuario["Tolerancia"]))
        arquivo.write("Melhor Elemento do Enxame={}\n".format(posicao_Gbest))
        arquivo.write("Valor na FO={0:.6e}\n".format(fitness_Gbest))
        arquivo.close()

        savetxt("Parametros_EstimadosPSO.txt",posicao_Gbest,delimiter='\t')
        savetxt("Valor FO PSO.txt",array([fitness_Gbest]))
        savetxt("ConvergenciaPSO.txt",convergencia,delimiter='\t')

        chdir(path.abspath(".."))
