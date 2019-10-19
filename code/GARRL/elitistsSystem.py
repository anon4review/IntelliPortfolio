from GARRL.rrl import RRL
import tensorflow as tf
import configparser
import os
import numpy as np
from GARRL.dataGenerator import DataGenerator
import tqdm

def getConfig(section, key):
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + '/config.ini'
    config.read(path,encoding='utf-8')
    return config.get(section, key)

class ElitistsSystem(object):
    def __init__(self):
        self.all_num=int(getConfig('elitists','all_num'))
        self.choosen_num = int(getConfig('elitists', 'choosen_num'))
        self.choosen_stocks_num=int(getConfig('rrl', 'choosen_stocks_num'))
        self.win_len=int(getConfig('rrl','window_len'))

        self.elitistsPool = []
        for _ in range(self.all_num):
            self.elitistsPool.append(RRL('rrl' + str(_)))
            print('elitist '+str(_)+' init completed!')
        print('elitists pool init complete!')
        self.choosenPool=[]
        self.datag=DataGenerator()

        #train
        self.train_epoch=int(getConfig('train','epoch'))
        self.train_num = int(getConfig('train', 'train_num'))
        self.one_step = int(getConfig('rrl', 'one_step'))

        self.sess=tf.Session()

    def gaSystemTrain(self,ga):
        train_data=self.datag.get_train_data(ga)
        train_index=self.datag.train_index
        index=0
        for elitist in self.elitistsPool:
            self.sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.VARIABLES, scope='rrl'+str(index))))
            for _ in tqdm.tqdm(range(self.train_epoch)):
                Ftminus1 = self.datag.Ftminus1
                _current_cell_state = np.zeros((1, self.choosen_stocks_num))
                _current_hidden_state=np.array([Ftminus1])
                for i in range(int((self.train_num-self.win_len+1)/self.one_step)):
                    feed = {elitist.input: train_data[:,i*self.one_step:self.one_step+self.one_step*i,:],
                            elitist.Ftminus1: Ftminus1,
                            elitist.price: self.datag.train_price[:,i*self.one_step:self.one_step+self.one_step*i],
                            elitist.cell_state:_current_cell_state,
                            elitist.hidden_state:_current_hidden_state,
                            elitist.index:train_index[i*self.one_step:self.one_step+self.one_step*i]
                            }
                    _, Ft,state = self.sess.run(
                        [elitist.train_step_noholder, elitist.origin_outputs,elitist.state], feed_dict=feed)
                    _current_cell_state, _current_hidden_state = state
                    Ftminus1=Ft[-1]
                _current_cell_state = np.zeros((1, self.choosen_stocks_num))
                _current_hidden_state = np.array([self.datag.Ftminus1])
                feed2 = {elitist.input: train_data,
                        elitist.Ftminus1: self.datag.Ftminus1,
                        elitist.price: self.datag.train_price,
                        elitist.cell_state: _current_cell_state,
                        elitist.hidden_state: _current_hidden_state,
                        elitist.index:train_index
                        }
                _,sharpe_ratio,ER,TE = self.sess.run([elitist.train_step_noholder,elitist.sharpe_ratio,elitist.ER,elitist.TE], feed_dict=feed2)
                print('sharpe_ratio:', str(sharpe_ratio))
                print('ER',ER)
                print('TE',TE)
            index+=1

    def choosen_elitists(self,ga):
        SR_list=[]
        eva_data=self.datag.get_eva_data(ga)
        _current_cell_state = np.zeros((1, self.choosen_stocks_num))
        _current_hidden_state = np.array([self.datag.Ftminus1])
        for elitist in self.elitistsPool:
            feed = {elitist.input: eva_data,
                    elitist.Ftminus1: self.datag.Ftminus1,
                    elitist.price:self.datag.eva_price,
                    elitist.cell_state: _current_cell_state,
                    elitist.hidden_state: _current_hidden_state
                    }
            SR = self.sess.run(elitist.sharpe_ratio, feed_dict=feed)
            SR_list.append(SR)
        #sort top 10
        #bubbleSort
        for i in range(self.elitistsPool.__len__() - 1):  # 这个循环负责设置冒泡排序进行的次数
            for j in range(self.elitistsPool.__len__() - i - 1):  # ｊ为列表下标
                if SR_list[j] < SR_list[j + 1]:
                    self.elitistsPool[j], self.elitistsPool[j + 1] = self.elitistsPool[j + 1], self.elitistsPool[j]
                    SR_list[j],SR_list[j+1]=SR_list[j+1],SR_list[j]

        self.choosenPool=self.elitistsPool[0:self.choosen_num]
        self.SR_list_choosen=SR_list[0:self.choosen_num]

    def get_fitness(self,ga):
        self.gaSystemTrain(ga)
        self.choosen_elitists(ga)
        fitness=1+np.sum(self.SR_list_choosen)/float(self.choosen_num)
        return fitness