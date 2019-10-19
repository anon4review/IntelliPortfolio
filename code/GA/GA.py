import random
import pandas as pd
import Portfolio_GA.Data_Preprocess

class GA(object):
    def init(self,l,num,stocklist,market_capitalization_list,index_path):
    # each organisms has l stocks
    # num is the number of population
        self.l=l
        self.stocklist = stocklist
        self.market_capitalization_list = market_capitalization_list
        self.index_path = index_path
        pop=[]
        for i in range(num):
            r = [random.random() for i in range(1, l)]
            r.sort()
            r.insert(0, 0)
            r.insert(l, 1)
            individual = [r[i + 1] - r[i] for i in range(len(r) - 1)]
            pop.append(individual)
        print('GA初始化完成')
        return pop

    def mutation(self,pop,prob):
        # The mutation rate runs from 0.05 to 0.06
        # pop is the population
        # prob is the probility of mutation
        # the mutation is shift mutation
        new_pop=[]
        for individual in pop:
            r = random.random()
            if r < prob:
                temp = individual.copy()
                pos = random.randint(0, len(individual)-1)
                ele = individual[pos]
                temp.remove(ele)
                temp.insert(0,ele)
                temp = self.check_constraint(temp)
                new_pop.append(temp)
        print('变异完成')
        return new_pop

    def crossover(self,pop,prob):
        # The crossover rate runs from 0.5 to 0.8
        # pop is the population
        # prob is the probility of crossover
        # the crossover is single point crossover
        new_pop=[]
        for i in range(len(pop)-1):
            individual1 = pop[i]
            individual2 = pop[i+1]
            r = random.random()
            if r < prob:
                #crossover
                temp1 = individual1.copy()
                temp2 = individual2.copy()
                pos = random.randint(0, len(pop)-1)
                temp2[0:pos] = individual1[0:pos]
                temp1[0:pos] = individual2[0:pos]
                #check the constraint
                temp1 = self.check_constraint(temp1)
                temp2 = self.check_constraint(temp2)
                new_pop.append(temp1)
                new_pop.append(temp2)
        print('交叉完成')
        return new_pop

    def check_constraint(self,individual):
        #Each organism has l gene points,the sum of all the gene points is 1.
        #After mutation or crossover, the organism maybe more or less  than 1.
        #This function is to check and regulate to satisfy the constraint.
        #确保权重都为正
        new_individual = individual.copy()
        for i in range(len(individual)):
            if individual[i] < 0:
                new_individual[i] = 0
        #确保权重之和为1
        sum_ = sum(new_individual)
        if sum_>1:
            while(sum_>1):
                max_ = max(new_individual)
                new_individual[new_individual.index(max(new_individual))] = 0
                sum_ = sum_ - max_
            if sum_ < 1:
                less = 1 - sum_
                i = new_individual.index(min(new_individual))
                new_individual[i] += less
                sum_ += less
            #print('>1')
        if sum_<1:
            less = 1-sum_
            i = new_individual.index(min(new_individual))
            new_individual[i] += less
            #print('<1')
        return new_individual

    def fitness(self,individual,stock_list,market_capitalization_list,index_path):
        #caculate the fitness of the portfolio by fomulation (4)
        sum_=0
        for stock in stock_list:
            sigma = self.caculate_sigma_k_2(stock,index_path)
            w_k = individual[stock_list.index(stock)]
            w_k_m = market_capitalization_list[stock_list.index(stock)]
            sum_ += (w_k-w_k_m)**2*sigma
        return 1/sum_

    def caculate_sigma_k_2(self,stock,index_path):
        #caculate the (sigma_k)^2 of the stock by fomulation
        #分子
        data1 = pd.read_excel(str(stock)+"_train.xls")
        molecule = data1['ReturnRate'].drop(index=0).var()
        #分母
        data2 = pd.read_excel(index_path)
        denominator = data2['ReturnRate'].drop(index=0).var()*(data2.shape[0]-1)
        return molecule/denominator

    def roulette(self,pop,selectNum):
        #Use roulette based on fitness
        popselect=[]
        fitTotal = 0
        fitlist = []
        for individual in pop:
            fit = self.fitness(individual = individual,stock_list = self.stocklist,market_capitalization_list = self.market_capitalization_list,index_path = self.index_path)
            fitlist.append(fit)
            fitTotal += fit
        print('个体适应度计算完成')

        #caculate cumulative fitness
        fitnesslist=[]
        fitnesslist.append(0.0)
        fitness=0.0
        for element in fitlist:
            fitness += element/fitTotal
            fitnesslist.append(fitness)
        fitnesslist[len(fitnesslist) -1] = 1.0

        #Elite retention 精英保留
        max_fitness = max(fitlist)
        popselect.append(pop[fitlist.index(max_fitness)])
        print('精英保留完成')

        for i in range(selectNum-1):
            r = random.random()
            for j in range(len(fitnesslist)-2):
                if (r > fitnesslist[j]) & (r < fitnesslist[j+1]):
                    popselect.append(pop[j+1])
        print('轮盘赌完成')
        print(popselect[0])
        print(sum(popselect[0]))
        return popselect,max_fitness,popselect[0]

class Portfolio(object):
    #given n stocks
    #return l stocks with different weights
    def init(self,l):
        self.l = l
        self.industry_dic,self.companylist = Portfolio_GA.Data_Preprocess.read_excel_industry_division(path1='E:\Portfolio_GA\上证120天数据\上证180行业划分.xlsx')
        self.mean_mc_list, self.mean_amount_list,self.w_m = self.caculate_amount_mc_scaled()

    def caculate_amount_mc_scaled(self):
        #caculate the mean traing amount and market capitalization of each company
        #scale the list of the mean
        mean_amount_list=[]
        mean_mc_list=[]
        for company in self.companylist:
            data = pd.read_excel(str(company) + "_train.xls")
            mean_amount = data['vol'].mean()
            mean_mc = data['TotalValue'].mean()
            mean_amount_list.append(mean_amount)
            mean_mc_list.append(mean_mc)
        #各个公司市值均值站市场总市值的比例，ga的适应度函数需要用到
        w_m = []
        sum_ = sum(mean_mc_list)
        for mean_mc in mean_mc_list:
            w_m.append(mean_mc/sum_)
        #scale
        max1 = max(mean_amount_list)
        min1 = min(mean_amount_list)
        for element in mean_amount_list:
            mean_mc_list[mean_amount_list.index(element)] = (element-min1)/(max1-min1)
        max2 = max(mean_mc_list)
        min2 = min(mean_mc_list)
        for element in mean_mc_list:
            mean_mc_list[mean_mc_list.index(element)] = (element-min2)/(max2 - min2)
        print('市值、交易量均值计算及归一化完成')
        return mean_mc_list,mean_amount_list,w_m

    def caculate_industry_market_capitalization(self):
        # caculate industry market capitalization
        market_capitalization_list = []
        for industry in self.industry_dic:
            company_list = self.industry_dic[industry]
            market_capitalization = 0
            for company in company_list:
                data = pd.read_excel(str(company) + "_train.xls")
                #使用市值的均值
                mc = data['TotalValue'].mean()
                market_capitalization += mc
            market_capitalization_list.append(market_capitalization)
        print('各行业总市值获取完成')
        return market_capitalization_list

    def capitalization_industry_max(self):
        #return the position of the max
        market_capitalization_list = self.caculate_industry_market_capitalization()
        max_position = market_capitalization_list.index(max(market_capitalization_list))
        print('市值最大行业获取完成')
        return max_position

    def judge_stock_select(self,v1,v2,v3):
        #After having the industry sector having the largest amount of market capitalization,we can select the company in this industry sector
        #select the stock having the highest P_ij to add to the portfolio.
        position = self.capitalization_industry_max()
        part_company_list = list(self.industry_dic.values())[position]
        max_P_ij = 0
        max = part_company_list[0]
        for company in part_company_list:
            data = pd.read_excel(str(company)+'_train.xls')
            data1 = pd.read_excel('index_train.xls')
            A = self.mean_amount_list[self.companylist.index(company)]
            M = self.mean_mc_list[self.companylist.index(company)]
            B = (data['ReturnRate'].drop(index=0).std())/((data1['ReturnRate'].drop(index=0).var()*(data1.shape[0]-1))**0.5)
            P_ij = v1*(1/B)+v2*A+v3*M
            if(P_ij > max_P_ij):
                max = company
        print('最大行业选股完成')
        return max

    def update(self,select_stock):
        # remove the stock selected from the industry sector
        # update the sort of market capitalization by industry
        industry_dic={}
        for industry in self.industry_dic:
            if select_stock in self.industry_dic[industry]:
                list1 = self.industry_dic[industry].copy()
                list1.remove(select_stock)
                industry_dic[industry] = list1
            else:
                industry_dic[industry] = self.industry_dic[industry]
        self.industry_dic = industry_dic
        print('行业字典更新完成')

    def select_l_stocks(self,v1,v2,v3):
        #caculate and select stock,then update the industry_dic until the portfolio have l stocks
        portfolio=[]
        market_capitalization_list = []
        for i in range(self.l):
            select_stock = self.judge_stock_select(v1,v2,v3)
            portfolio.append(select_stock)
            market_capitalization_list.append(self.w_m[self.companylist.index(select_stock)])
            self.update(select_stock)
            print('第'+str(i+1)+'支股票选取完成：'+select_stock)
        print(str(self.l)+'支股票选取完成')
        return portfolio,market_capitalization_list

if __name__=='__main__':
    port = Portfolio()
    port.init(l = 10)
    portfolio,mc_list = port.select_l_stocks(v1=2,v2=1,v3=1)
    print(portfolio)
    ga = GA()
    pop = ga.init(l=10,num=50,stocklist=portfolio,market_capitalization_list = mc_list,index_path='index_train.xls')
    j = 0
    last_fit = -1
    for i in range(1000000):
        mutation_pop = ga.mutation(pop=pop,prob=0.05)
        crossover_pop = ga.crossover(pop=pop,prob=0.6)
        pop.extend(mutation_pop)
        pop.extend(crossover_pop)
        pop_,best_fitness,best = ga.roulette(pop=pop,selectNum=50)
        pop = pop_
        print(i)
        print(best_fitness)
        if(best_fitness == last_fit):
            j+=1
        else:
            j = 0
        last_fit = best_fitness
        #save the result
        dic = {}
        for i in range(len(portfolio)):
            dic[portfolio[i]] = best[i]
            print(str(portfolio[i]) + ":" + str(best[i]))
        print(dic)
        df = pd.DataFrame(best, portfolio)
        df.to_excel('result.xls')
        if(j>5000):
            break
