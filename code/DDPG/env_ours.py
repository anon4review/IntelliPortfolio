import numpy as np
import sqlite3
from Portfolio_ddpg.read_config import *
import math

def read_data():
    codelist = []
    table_name=config.get('db','table_name')
    db_name = config.get('db', 'db_name')
    conn = sqlite3.connect(db_name)
    SQL = "select distinct code from %s" %table_name
    cursor = conn.execute(SQL)
    for row in cursor:
        codelist.append(row[0])

    stock_choose_num = int(config.get('data', 'stock_choose_num'))
    date = int(config.get('data','all_days_used'))
    train_num = int(config.get('data','train_num'))
    feature_num = int(config.get('data','feature_num'))
    result = np.array([[[0.0 for _ in range(feature_num+1)]for _ in range(date)] for _ in range(stock_choose_num)])
    stock_i = 0
    for code in codelist:
        SQL = 'select * from %s where code=\'%s\'' % (table_name,code)
        cursor = conn.execute(SQL)
        date_i = 0
        for row in cursor:
            result[stock_i][date_i] = row[2:]
            date_i += 1
        stock_i += 1


    date_i=0
    index=np.array([0.0 for _ in range(date)])
    SQL = 'select closeprice from indexes'
    cursor = conn.execute(SQL)
    for row in cursor:
        index[date_i] = row[0]
        date_i+=1
    return result[:,:train_num,:],result[:,train_num:,:],index,codelist

class Env(object):
    def __init__(self):
        self.data,self.test_data,self.index,self.codelist = read_data()
        self.win_len = int(config.get('ddpg','win_len'))
        self.stock_num = int(config.get('data','stock_choose_num'))
        self.feature_num = int(config.get('data','feature_num'))
        #observation
        self.state_dim = self.win_len*self.stock_num*self.feature_num+self.stock_num
        #self.state_dim = self.win_len * self.stock_num + self.stock_num
        self.state = self.win_len-1 #标记当前状态的索引
        #action
        self.action_dim = self.stock_num
        #action_bound是为了调节输出动作范围和真实动作范围之间的scale差距
        # tanh输出是【-1，1】，我们需要的是【0，1】，因此在网络的输出部分不需要调整scale，
        self.action_bound = [[1.] for i in range(self.stock_num)]
        self.epsilon = 1e-5

    def step(self,state,action_):
        s_ = np.zeros((1,self.state_dim), dtype=np.float32)
        done = True
        #check the legality of the action
        if(sum(action_)>1):
            r = 0
        else:
            #state+1之前各股票的价格
            value = self.data[:,self.state,self.feature_num:]
            #前一天的动作
            action = state[:,-self.stock_num:]
            while(0 in value):
                value[value.index(0)] += self.epsilon
            index = self.index[self.state]
            if(index == 0):
                index += self.epsilon
            self.state +=1
            if(self.state == self.data.shape[1]):
                #避免越界，若到达数据最后，则state不变，reward为0
                self.reset()
                s_ = self.data[:, self.state - self.win_len-1:self.state-1, :self.feature_num].reshape((1,self.state_dim-self.stock_num))
                #s_ = self.data[:, self.state - self.win_len - 1:self.state - 1, self.feature_num:].reshape((1, self.state_dim - self.stock_num))
                init_weights = np.array([0.1 for _ in range(self.stock_num)]).reshape(1, self.stock_num)
                s_ = np.concatenate([s_,init_weights],axis=1)
                r = 0
                done = False
            else:
                s_ = self.data[:, self.state - self.win_len+1 :self.state+1, :self.feature_num].reshape((1,self.state_dim-self.stock_num))
                #s_ = self.data[:, self.state - self.win_len + 1:self.state + 1, self.feature_num:].reshape((1, self.state_dim - self.stock_num))
                s_ = np.concatenate([s_,action_.reshape(1,self.stock_num)],axis=1)
                #state+1之后的各股票价格
                value_ = self.data[:,self.state,self.feature_num:]
                index_ = self.index[self.state]
                #tracking error
                TE = math.fabs(np.log(np.matmul(action_,value_)/(np.matmul(action,value)+self.epsilon)+self.epsilon)-np.log(index_/index+self.epsilon))
                #excess return
                ER = np.log(np.matmul(np.array([value_[i]/value[i] for i in range(len(value))]).reshape(1,self.stock_num),action.reshape(self.stock_num,1))+self.epsilon)-np.log(index_/index+self.epsilon)

                rt = np.log(np.matmul(action_,value_)/(np.matmul(action,value)+self.epsilon)+self.epsilon)

                #r = np.log(ER + 1/TE + self.epsilon)
                #r = np.log(1/TE)
                #print(TE)
                #print(ER)
                r = (1-LAMDA)*ER - LAMDA*TE
                #if(ER>=0):
                #    r = ER/TE
                #else:
                #    r = -ER/TE
                #r = rt
        #print(r)
        return s_,r[0][0],done

    def reset(self):
        state = self.data[:,:self.win_len,:self.feature_num].reshape((1,self.state_dim-self.stock_num))
        #state = self.data[:, :self.win_len, self.feature_num:].reshape((1, self.state_dim - self.stock_num))
        init_weights = np.array([0.1 for _ in range(self.stock_num)]).reshape(1,self.stock_num)
        state = np.concatenate([state,init_weights],axis = 1)
        self.state = self.win_len - 1
        return state

if __name__ == '__main__':
    list=[1,2,3,4]
    print(list[:-1])

