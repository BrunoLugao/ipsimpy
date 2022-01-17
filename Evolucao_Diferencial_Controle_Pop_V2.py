# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 12:36:59 2018

@author: Bruno
"""
from numpy import array, zeros, random, append, copy, savetxt, mean, empty
import operator #Para ordenar uma classe de acordo com um atributo
from os import mkdir, path, chdir
from time import time
from sys import stdout


class opcoes:
    Usuario={"F":0.8,
            "CR":0.5,
            "Gmax":100,
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
        if type(self.Usuario["F"])!=float or self.Usuario["F"]<=0 or self.Usuario["F"]>2.0:
            raise Exception("O valor de F deve ser um real entre 0.0 e 2.0")

        if type(self.Usuario["CR"])!=float or self.Usuario["CR"]<0.0 or self.Usuario["CR"]>1.0:
            raise Exception("O valor de CR deve ser um real entre 0.0 e 1.0")

        if type(self.Usuario["Gmax"])!=int or self.Usuario["Gmax"]<=0:
            raise Exception("O valor de Gmax deve ser um inteiro maior que Zero")

        if type(self.Usuario["Tolerancia"])!=float:
            raise Exception("O valor da Tolerância deve ser um real")

        if type(self.Usuario["VerOpcoes"])!=bool:
            raise Exception("VerOpcoes deve ser True or False")

class Individuo(opcoes):
    def __init__(self,P0):
        self.posicao=array([])
        self.fitness=-1.0
        self.dimensao=len(P0)
        self.posicao=copy(P0)

    def Avaliar_Fitness(self,FO):
        self.fitness=FO(self.posicao)

    def Mutacao(self,indice,tamanho_populacao,populacao):
        r0,r1,r2=random.choice(tamanho_populacao, 3,replace=False)
        v0=copy(populacao[r0].posicao)
        v1=copy(populacao[r1].posicao)
        v2=copy(populacao[r2].posicao)
        vetor_mutante=copy(v0 + self.Usuario["F"]*(v2 - v1))
        return vetor_mutante

    def Cruzamento(self, v_i, limites):
        ui = zeros(self.dimensao)
        for i in range(self.dimensao):
            rj = random.uniform(0, 1)
            lj = random.randint(1, self.dimensao)

            if rj <= self.Usuario["CR"] or lj == i:
                ui[i] = copy(v_i[i])
            else:
                ui[i] = copy(self.posicao[i])

            if ui[i] < limites[i][0]:
                ui[i] = limites[i][0]

            if ui[i] > limites[i][1]:
                ui[i] = limites[i][1]
        return ui

    def Atualiza_Posicao(self,posicao,fitness):
        self.posicao=copy(posicao)
        self.fitness=fitness


class Populacao_Inicial:
    def __new__(cls,tamanho_pop,limit):
        cls.pop=array([])
        dim=len(limit)
        Populacao_Ini=zeros([tamanho_pop,dim])
        for i in range(tamanho_pop):
            for j in range(dim):
                Populacao_Ini[i][j]=random.uniform(limit[j][0],limit[j][1])

        for i in range(tamanho_pop):
            cls.pop=append(cls.pop,Individuo(Populacao_Ini[i]))

        return cls.pop

class Populacao_Iteracao:
    def __new__(cls,populacao,tamanho_pop,tamanho_old,limit):
        cls.pop=array([])
        dim=len(populacao[0].posicao)
        
        if tamanho_pop<tamanho_old:

            for i in range(tamanho_pop):
                cls.pop=append(cls.pop,populacao[i])#ACHO QUE TERIA QUE RANQUEAR OS INDIVIDUOS PARA NAO PERDER OS MELHORES
        
        else:
            diferenca=tamanho_pop-tamanho_old
            
            for i in range(tamanho_pop):
                cls.pop=append(cls.pop,populacao[i])
            
            Populacao_Nova=zeros([diferenca,dim])
        
            for i in range(diferenca):
                for j in range(dim):
                    Populacao_Nova[i][j]=random.uniform(limit[j][0],limit[j][1])

                cls.pop=append(cls.pop,Individuo(Populacao_Nova[i]))
            
        return cls.pop


class Evolucao_Diferencial_Controle_Pop_V2(Individuo):
    Defaut={"F":0.8,
            "CR":0.5,
            "Gmax":100,
            "Tolerancia":10**-5,
            "VerOpcoes":False}

    Convergencia=array([])
    N_populacao=array([])
    Parametros_Estimados=array([])
    Valor_FO=0.0
    Custo_Computacional=None

    def __init__(self,FuncaoObjetivo, Tamanho_Populacao_Min,Tamanho_Populacao_Max, limites, OpcoesDE=None):
        self.Definicao_Parametros_Controle(self.Defaut)
        self.Definicao_Parametros_Controle(OpcoesDE)
        self.Validacao_Parametros_Controle()


        posicao_best=array([])
        fitness_best=-1.0
        convergencia=array([])
        N_Populacao=array([])

        controle=0
        
        Tamanho_Populacao=Tamanho_Populacao_Max
        Tamanho_Populacao_Old=Tamanho_Populacao_Max
        
        Populacao=Populacao_Inicial(Tamanho_Populacao,limites)
        Populacao_Old=copy(Populacao)
        
        Populacao[0].Avaliar_Fitness(FuncaoObjetivo)
        posicao_best = copy(Populacao[0].posicao)
        fitness_best = Populacao[0].fitness

        g=0
        ini=time()
        while g<self.Usuario["Gmax"] and fitness_best>=self.Usuario["Tolerancia"]:
            vetor_fitness=zeros(Tamanho_Populacao)
            #print("Inicio")
            for i in range(Tamanho_Populacao):
                vetor_mutante    = copy(Populacao[i].Mutacao(i,Tamanho_Populacao,Populacao_Old))
                vetor_resultante = copy(Populacao[i].Cruzamento(vetor_mutante,limites))

                fitness_vr = FuncaoObjetivo(vetor_resultante)
                Populacao[i].Avaliar_Fitness(FuncaoObjetivo)
                vetor_fitness[i]=Populacao[i].fitness
                
                if fitness_vr < Populacao[i].fitness:
                    Populacao[i].Atualiza_Posicao(vetor_resultante,fitness_vr)

                if Populacao[i].fitness<fitness_best:
                    posicao_best = copy(Populacao[i].posicao)
                    fitness_best = Populacao[i].fitness
            
            Populacao= sorted(Populacao, key=operator.attrgetter('fitness'))
            
            convergencia = append(convergencia, fitness_best)
            
            omega=mean(vetor_fitness)/max(vetor_fitness)
            #print("Tamanho Pop",Tamanho_Populacao)
            
            N_Populacao = append(N_Populacao, Tamanho_Populacao)
            
            Tamanho_Populacao_Old=Tamanho_Populacao
            Tamanho_Populacao=int(round(Tamanho_Populacao_Min*omega + Tamanho_Populacao_Max*(1-omega)))
            #print("fitness:",vetor_fitness)

            if Tamanho_Populacao == Tamanho_Populacao_Old:
                controle=controle+1

                if controle == 10:
                    Tamanho_Populacao=Tamanho_Populacao_Max
                    controle=0
           
            #print("Tamanho Pop Novo",Tamanho_Populacao)
            Populacao_Old=Populacao_Iteracao(Populacao,Tamanho_Populacao,Tamanho_Populacao_Old,limites)
            
            stdout.write("\rGeração: %d" % (g+1))
            stdout.flush()
            g=g+1

        fim=time()
        #stdout.write("\n")
        
        self.Convergencia = copy(convergencia)
        self.N_Populacao = copy(N_Populacao)
        self.Parametros_Estimados = copy(posicao_best)
        self.Valor_FO = fitness_best
        self.Custo_Computacional=fim-ini

        if self.Usuario["VerOpcoes"] == True:
            print("")
            print("OPÇÕES EVOLUÇÃO DIFERENCIAL")
            print("Tamanho População={}".format(Tamanho_Populacao))
            print("Numero Máximo de Gerações={}".format(self.Usuario["Gmax"]))
            print("Numero de Gerações Utilizado={}".format(g))
            print("Taxa de Mutacao={}".format(self.Usuario["F"]))
            print("Taxa de CrossOver={}".format(self.Usuario["CR"]))
            print("Melhor Membro da População={}".format(posicao_best))
            print("Valor na FO={0:.6e}".format(fitness_best))
            print("Tolerância={0:.0e}".format(self.Usuario["Tolerancia"]))
            print("")

        Diretorio_Resultados="Arquivos Saida ED Pop"

        if path.isdir(Diretorio_Resultados):
            chdir(path.abspath(Diretorio_Resultados))
        else:
            mkdir(Diretorio_Resultados)
            chdir(path.abspath(Diretorio_Resultados))

        arquivo=open("DetalhesED.txt","w")
        arquivo.write("OPÇÕES EVOLUÇÃO DIFERENCIAL\n")
        arquivo.write("Tamanho População={}\n".format(Tamanho_Populacao))
        arquivo.write("Numero Máximo de Gerações={}\n".format(self.Usuario["Gmax"]))
        arquivo.write("Numero de Gerações Utilizado={}\n".format(g))
        arquivo.write("Taxa de Mutacao={}\n".format(self.Usuario["F"]))
        arquivo.write("Taxa de CrossOver={}\n".format(self.Usuario["CR"]))
        arquivo.write("Melhor Membro da População={}\n".format(posicao_best))
        arquivo.write("Valor na FO={0:.6e}\n".format(fitness_best))
        arquivo.write("Tolerância={0:.0e}\n".format(self.Usuario["Tolerancia"]))
        arquivo.close()

        savetxt("Parametros_EstimadosED_Pop.txt",posicao_best,delimiter='\t')
        savetxt("Valor FO ED Pop.txt",array([fitness_best]))
        savetxt("ConvergenciaED_Pop.txt",convergencia,delimiter='\t')
        savetxt("N_Pop.txt",N_Populacao,delimiter='\t')

        chdir(path.abspath(".."))
