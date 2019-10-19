from heuristicGA.dataKeeper import DataKeeper
from configparser import ConfigParser
import numpy as np
import random
from heuristicGA.chromosome import Chromosome
class GA(object):
    def __init__(self):
        '''
        origin只有revise mode时有用，传进来的是一个各股票占C的比例，需要转换成如下形式：
        revise：Q:Q
                s:[0,origin],代表投资组合不变的solution
                y:s2y
                fitness
                unfitness

        '''
        cp = ConfigParser()
        cp.read('config.conf')
        self.alpha = float(cp.get('hy', 'alpha'))
        self.lamda = float(cp.get('hy', 'lamda'))
        self.T = int(cp.get('hy', 'T'))
        self.L = int(cp.get('hy', 'L'))
        self.K = int(cp.get('hy', 'K'))
        self.N = int(cp.get('hy', 'N'))
        self.mode = cp.get('ga', 'mode')
        self.win_len = int(cp.get('hy', 'win_len'))
        self.loopN = int(cp.get('ga', 'loopN'))
        self.gamma=float(cp.get('hy', 'gamma'))
        self.epsilon = float(cp.get('hy', 'epsilon'))
        self.delta=float(cp.get('hy', 'delta'))
        self.population_size = int(cp.get('ga', 'population_size'))
        self.C = float(cp.get('ga', 'C'))
        self.origin = list(map(float, cp.get('ga', 'origin').split(',')))
        self.y = [0.0 for i in range(self.N + 1)]
        if(sum(self.origin)!=1):
            print('sum of origin not 1!')
            exit(-4)
        self.Q = list(map(int, cp.get('ga', 'Q').split(',')))

        if(min(self.Q)==0):
            print('Q begin at 1 not 0!')
            exit(-5)

        if (len(self.Q) != len(self.origin)):
            print('Q and origin shape is wrong!')
            exit(-1)

        self.datakeeper = DataKeeper()
        self.data=self.datakeeper.data

        if (self.mode == 'revise'):
            index = 0
            for i in self.Q:
                self.y[i] = self.origin[index]
                index += 1
        if (self.mode == 'create'):
            self.Q = [i for i in range(1, self.K + 1)]
            for i in self.Q:
                self.y[i] = 1.0/self.K
        self.decision = []
        self.decision.append(self.y[1:])



    def ga_loop(self):
        for date in range(self.L):
            self.TV = self.data[:, self.T+date-self.win_len+1:self.T+date]
            self.TVbefore = self.data[:, self.T+date-self.win_len:self.T+date-1]

            self.index = self.datakeeper.index
            self.TI = self.index[self.T+date-self.win_len+1:self.T+date]
            self.TIbefore = self.index[self.T+date-self.win_len:self.T+date-1]
            self.best_chrom = Chromosome()
            self.population_init()
            print('-------------------------population init complete--------------------------------')
            for i in range(self.loopN):
                parent1,parent2=self.choose_parent()
                child=self.make_children(parent1,parent2,i)
                self.set_best_chrom(child)
                self.choose_kill(child)
            #最后一天的decision无用

            self.decision.append(self.best_chrom.y[1:])
            self.y=self.best_chrom.y
            self.Q=self.best_chrom.Q
            print('-------------------------ga complete--------------------------------')



    def population_init(self):
        if (not(self.mode == 'revise' or self.mode=='create')):
            print('Wrong mode! Please choose from create or revise!')
            exit(-2)

        self.population = []
        self.num_to_gen = self.population_size-1

        chrom = Chromosome()
        chrom.Q = self.Q
        chrom.y = self.y
        s = self.y2s(chrom)
        chrom.s = s
        chrom.fitness = self.calculate_fitness(chrom)
        chrom.unfitness = self.calculate_unfitness(chrom)

        self.population.append(chrom)
        self.set_best_chrom(chrom)

        for k in range(self.num_to_gen):
            chrom=Chromosome()
            Q=random.sample(range(1,self.N+1),self.K)
            chrom.Q=Q
            code = np.random.random(self.K + 1)
            chrom.s = code
            y = self.s2y(chrom)
            chrom.y = y
            chrom.fitness = self.calculate_fitness(chrom)
            chrom.unfitness = self.calculate_unfitness(chrom)
            self.population.append(chrom)
            self.set_best_chrom(chrom)


    def if_feasible(self,chrom):
        if chrom.unfitness<=0.001*self.gamma:
            return True
        else:
            return False

    def sys_out(self):
        print('fitness:', self.best_chrom.fitness)
        print('unfitness:', self.best_chrom.unfitness)

    def set_best_chrom(self,chrom):
        if (self.best_chrom .Q.__len__()== 0):
            #if self.if_feasible(chrom):
                self.best_chrom.Q = chrom.Q
                self.best_chrom.s=chrom.s
                self.best_chrom.y=chrom.y
                self.best_chrom.fitness=chrom.fitness
                self.best_chrom.unfitness=chrom.unfitness
                self.sys_out()
        elif(self.best_chrom.fitness>chrom.fitness and self.if_feasible(chrom) ):
            self.best_chrom.Q = chrom.Q
            self.best_chrom.s = chrom.s
            self.best_chrom.y = chrom.y
            self.best_chrom.fitness = chrom.fitness
            self.best_chrom.unfitness = chrom.unfitness
            self.sys_out()

    def choose_parent(self):
        '''binary tournament selection'''
        rand1=random.randint(0,self.population_size-1)
        rand2 = random.randint(0, self.population_size - 1)
        pool1_pa1_fitness=self.population[rand1].fitness
        pool1_pa2_fitness =self.population[rand2].fitness
        parent1=self.population[rand1] if pool1_pa1_fitness<pool1_pa2_fitness else self.population[rand2]
        rand3 = random.randint(0, self.population_size - 1)
        rand4 = random.randint(0, self.population_size - 1)
        pool2_pa1_fitness = self.population[rand3].fitness
        pool2_pa2_fitness = self.population[rand4].fitness
        parent2 = self.population[rand3] if pool2_pa1_fitness < pool2_pa2_fitness else self.population[rand4]

        return parent1,parent2

    def calculate_fitness(self,chromosome):
        y=np.array(chromosome.y[1:]).reshape(1,-1)
        VT=np.array(self.TV[:,-1]).reshape((-1,1))
        tracing_error = np.power(np.sum(np.power(np.abs(np.log(np.divide(np.matmul(y,np.divide(self.TV,VT)),np.matmul(y,np.divide(self.TVbefore,VT)))) - np.divide(self.TI,self.TIbefore)),self.alpha)),1.0/self.alpha)/self.T
        excess_return=np.sum(np.log(np.divide(np.matmul(y,np.divide(self.TV,VT)),np.matmul(y,np.divide(self.TVbefore,VT))))- np.divide(self.TI,self.TIbefore))/self.T
        fitness=self.lamda*tracing_error-(1-self.lamda)*excess_return
        #print(fitness)
        return fitness

    def F(self,X,CyDivV):
        result=0.01*np.matmul(np.abs(X-CyDivV),np.transpose(self.TV[:,-1]))
        return result

    def calculate_unfitness(self,chromosome):
        X=np.divide(np.multiply(self.C,self.y[1:]),self.TV[:,-1])
        CyDivV=np.divide(np.multiply(self.C,chromosome.y[1:]),self.TV[:,-1])
        #print(chromosome.y[0])
        #print(self.F(X,CyDivV)/self.C)
        unfitness=np.abs(chromosome.y[0]-self.F(X,CyDivV)/self.C)
        #print(unfitness)
        #print('--------------------------------')
        return unfitness

    def s2y(self,chromosome):
        epsilon=[0.0 for i in range(self.N+1)]
        for _ in chromosome.Q:
            epsilon[_]=self.epsilon

        delta=[self.delta for i in range(self.N+1)]
        delta[0]=self.gamma

        Q=[0]
        for i in chromosome.Q:
            Q.append(i)
        R=[]

        epsilon_sum=sum(epsilon)
        s_sum=sum(chromosome.s)
        if(epsilon_sum>1):
            print('epsilon sum bigger than 1')
            exit(-3)
        s=[0.0 for i in range(self.N+1)]
        for _ in chromosome.Q:
            index=chromosome.Q.index(_)
            s[_]=chromosome.s[index+1]
        s[0]=chromosome.s[0]
        y=epsilon+np.multiply(s,(1-epsilon_sum)/s_sum)

        QminusR = [item for item in Q if not item in R]
        ifLoop=True
        while(ifLoop):
            ifLoop=False
            for i in QminusR:
                if y[i]>=delta[i]:
                    ifLoop=True
                    R.append(i)

            QminusR = [item for item in Q if not item in R]
            epsilon_sum = 0
            delta_sum = 0
            s_sum = 0
            for _ in QminusR:
                epsilon_sum += epsilon[_]
                s_sum += s[_]
            for _ in R:
                delta_sum += delta[_]
            for k in QminusR:
                y[k] = epsilon[k] + s[k] * (1 - epsilon_sum-delta_sum) / s_sum
            for k in R:
                y[k]=delta[k]

        return y

    def y2s(self,chromosome):
        '''这个算法目前是不对的，暂时没想出来如何根据y计算出s'''
        epsilon_last=self.epsilon
        y_last=chromosome.y[chromosome.Q[-1]]
        epsilon = np.array([0.0 for i in range(self.N + 1)])
        for _ in chromosome.Q:
            epsilon[_] = self.epsilon
        sum_epsilon=sum(epsilon)

        x=np.divide(y_last-epsilon_last,1-sum_epsilon)#s/s_sum=x
        s=[0.0 for i in range(chromosome.Q.__len__()+1)]
        s_sum=0
        for i in range(chromosome.Q.__len__()-1):
            rand=random.random()
            s[1+i]=rand
            s_sum+=rand
        s_last=np.divide(x*s_sum,1-x)
        s[-1]=s_last
        return s


    def make_children(self,parent1,parent2,step):
        num2mul = 0.005
        #num2mul=0.3-step*0.0001
        #num2mul=0.005 if num2mul<0.005 else num2mul
        child=Chromosome()
        rand = random.random()
        if rand < 0.5:
            child.s.append(parent1.s[0])
        else:
            child.s.append(parent2.s[0])

        #crossover
        for stock in parent1.Q:
            if(stock in parent2.Q):
                child.Q.append(stock)
                rand=random.random()
                if rand<0.5:
                    s_index=parent1.Q.index(stock)
                    child.s.append(parent1.s[s_index+1])
                else:
                    s_index = parent2.Q.index(stock)
                    child.s.append(parent2.s[s_index + 1])
            else:
                rand = random.random()
                if rand < 0.5:
                    child.Q.append(stock)
                    s_index = parent1.Q.index(stock)
                    child.s.append(parent1.s[s_index + 1])
        for stock in parent2.Q:
            if(stock not in parent1.Q):
                rand = random.random()
                if rand < 0.5:
                    child.Q.append(stock)
                    s_index = parent2.Q.index(stock)
                    child.s.append(parent2.s[s_index + 1])


        #mutate
        rand_stock=random.randint(0,self.N)
        if rand_stock in child.Q :
            s_index = child.Q.index(rand_stock)
            rand = random.random()
            if rand < 0.5:
                child.s[s_index + 1]+=child.s[s_index + 1]*num2mul
            else:
                child.s[s_index + 1] -= child.s[s_index + 1]*num2mul
        elif rand_stock==0:
            rand = random.random()
            if rand < 0.5:
                child.s[0] += child.s[0] * num2mul
            else:
                child.s[0] -= child.s[0] * num2mul
        else:
            child.Q.append(rand_stock)
            rand=random.random()
            child.s.append(rand)

        self.make_child_exact_k(child)

        child.y=self.s2y(child)
        child.fitness=self.calculate_fitness(child)
        child.unfitness=self.calculate_unfitness(child)
        return child

    def make_child_exact_k(self,child):
        num=self.K-child.Q.__len__()
        if(num<0):
            stocks=random.sample(child.Q,self.K)
            QminusStocks=[item for item in child.Q if not item in stocks]
            for i in QminusStocks:
                index=child.Q.index(i)
                del child.Q[index]
                del child.s[index+1]
        elif(num>0):
            while(num!=0):
                rand_stock = random.randint(1, self.N)
                if not rand_stock in child.Q:
                    child.Q.append(rand_stock)
                    rand=random.random()
                    child.s.append(rand)
                    num-=1

    def choose_kill(self,child):
        fitness=child.fitness
        unfitness=child.unfitness
        G2tokill=-1
        G3tokill = -1
        G4tokill = -1
        for i in range(self.population.__len__()):
            father=self.population[i]
            if(father.fitness>=fitness and father.unfitness>=unfitness):
                del self.population[i]
                self.population.append(child)
                return
            elif (father.fitness <= fitness and father.unfitness >= unfitness):
                if G2tokill==-1 or self.population[G2tokill].unfitness<father.unfitness:
                    G2tokill=i
            elif(father.fitness >= fitness and father.unfitness <= unfitness):
                if G3tokill==-1 or self.population[G3tokill].unfitness<father.unfitness:
                    G3tokill=i
            elif (father.fitness <= fitness and father.unfitness <= unfitness):
                if G4tokill==-1 or self.population[G4tokill].unfitness<father.unfitness and self.population[G4tokill].fitness<father.fitness:
                    G4tokill=i
        if(G2tokill!=-1):
            del self.population[G2tokill]
        elif(G3tokill!=-1):
            del self.population[G3tokill]
        elif (G4tokill != -1):
            del self.population[G4tokill]
        else:
            print('choose to kill function is wrong!')
            exit(-6)
        self.population.append(child)










if __name__=="__main__":
    ga=GA()