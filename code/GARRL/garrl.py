import tensorflow as tf
import numpy as np
import configparser
import os
import random
from rrl import RRL
from elitistsSystem import ElitistsSystem

def getConfig(section, key):
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + '/config.ini'
    config.read(path,encoding='utf-8')
    return config.get(section, key)

class GARRL(object):
    def __init__(self):
        self.population_size = int(getConfig('ga', 'population_size'))
        self.generation_num = int(getConfig('ga', 'generation_num'))
        self.crossover_prob = float(getConfig('ga', 'crossover_prob'))
        self.mutation_prob = float(getConfig('ga', 'mutation_prob'))

        self.indicator_num = int(getConfig('rrl', 'indicator_num'))
        self.avg_elitist = RRL(scope='avg_elitist')
        self.elitistsSystem=ElitistsSystem()
        self.population=[]
        self.fitness=[]

        self.best_fitness=-10000
        self.best_indicator_type=0
        self.best_choosen_elistists=[]

        self.saver = tf.train.Saver()
        self.sess=self.elitistsSystem.sess
        self.sess.run(tf.initialize_variables(self.avg_elitist.vars))
        print('ga population initing')
        self.population_init()

    def note_get_better_indicator_type(self):
        print('get better indicator type!')
        print('indicator type:',self.best_indicator_type)
        print('fitness(SR_):',self.best_fitness)

    def get_fitness(self,chrom):
        fitness=self.elitistsSystem.get_fitness(chrom)
        return fitness

    def population_init(self):
        for i in range(self.population_size):
            chrom=[]
            for j in range(self.indicator_num):
                chrom.append(random.randint(0,1))
            fitness=self.get_fitness(chrom)
            self.fitness.append(fitness)
            self.population.append(chrom)
            if(self.best_fitness<=fitness):
                self.best_fitness=fitness
                self.best_indicator_type=chrom
                self.note_get_better_indicator_type()
                self.best_choosen_elistists=self.elitistsSystem.choosenPool.copy()
                self.saver.save(self.sess, './best_elis-model')
            print('population '+str(i)+':'+str(chrom))
            print('population ' + str(i) + ' fitness(SR_):' + str(fitness))
            print('========================================================')
        print('population init complete!')
        print('========================================================')

    def roulette_wheel_selection(self):
        sum_fitness=sum(self.fitness)
        prob=np.array(self.fitness)/sum_fitness
        wheel_prob=[]
        for i in range(len(prob)):
            wheel_prob.append(sum(prob[0:i+1]))
        rand1=random.random()
        for j in range(wheel_prob.__len__()):
            if wheel_prob[j]>=rand1:
                self.father1=self.population[j]
                self.father1_index=j
                break
        rand2 = random.random()
        for j in range(wheel_prob.__len__()):
            if wheel_prob[j] >= rand2:
                self.father2 = self.population[j]
                self.father2_index=j
                break

    def uniform_crossover(self):
        child1=[]
        child2=[]
        for i in range(self.indicator_num):
            rand=random.random()
            if rand<=self.crossover_prob:
                child1.append(self.father2[i])
                child2.append(self.father1[i])
            else:
                child1.append(self.father1[i])
                child2.append(self.father2[i])
        return child1,child2
    def mutate(self,child):
        rand=random.random()
        if(rand<=self.mutation_prob):
            rand_index=random.randint(0,self.indicator_num-1)
            child[rand_index]=abs(child[rand_index]-1)

    def ga(self):
        for epoch in range(self.generation_num):
            #choose two father
            self.roulette_wheel_selection()
            child1,child2=self.uniform_crossover()
            self.mutate(child1)
            fitness1 = self.get_fitness(child1)
            print('child ' + ':' + str(child1))
            print('child ' + 'fitness(SR_):' + str(fitness1))
            print('========================================================')
            if (self.best_fitness <= fitness1):
                self.best_fitness = fitness1
                self.best_indicator_type = child1
                self.note_get_better_indicator_type()
                self.best_choosen_elistists = self.elitistsSystem.choosenPool.copy()
                self.saver.save(self.sess, './best_elis-model')

            self.mutate(child2)
            fitness2 = self.get_fitness(child2)
            print('child ' + ':' + str(child1))
            print('child ' + 'fitness(SR_):' + str(fitness1))
            print('========================================================')
            if (self.best_fitness <= fitness2):
                self.best_fitness = fitness2
                self.best_indicator_type = child2
                self.note_get_better_indicator_type()
                self.best_choosen_elistists = self.elitistsSystem.choosenPool.copy()
                self.saver.save(self.sess, './best_elis-model')

            if(fitness1>self.fitness[self.father1_index]):
                self.population[self.father1_index]=child1
                self.fitness[self.father1_index]=fitness1

            if (fitness2 > self.fitness[self.father2_index]):
                self.population[self.father2_index] = child2
                self.fitness[self.father2_index] = fitness2