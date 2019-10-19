from heuristicGA.chromosome import *
import numpy as np
from heuristicGA.dataKeeper import *
from configparser import ConfigParser
from numpy import *
#set_printoptions(threshold=NaN)
class TLTest(object):
    def __init__(self):
        cp = ConfigParser()
        cp.read('config.conf')
        self.alpha = float(cp.get('hy', 'alpha'))
        self.lamda = float(cp.get('hy', 'lamda'))
        self.T = int(cp.get('hy', 'T'))
        self.L = int(cp.get('hy', 'L'))
        self.cash = float(cp.get('ga', 'C'))

        self.gamma = float(cp.get('hy', 'gamma'))
        self.epsilon = float(cp.get('hy', 'epsilon'))
        self.delta = float(cp.get('hy', 'delta'))

        self.C = float(cp.get('ga', 'C'))
        self.origin = list(map(float, cp.get('ga', 'origin').split(',')))

        datakeeper = DataKeeper()
        self.data = np.array(datakeeper.data)
        self.price = self.data[:, self.T:self.T+self.L]
        self.index = np.array(datakeeper.index)
        self.index=self.index[self.T:self.T+self.L]
        self.all_market_capitalization=datakeeper.all_market_capitalization[:,self.T:self.T+self.L]

    def get_L_object(self,decisions):
        y = decisions[0:-1]
        y_new=decisions[1:]
        price_before = np.transpose(self.price[:, 0:-1])  # [step,stock_num] 从第0天开始
        price_later = np.transpose(self.price[:, 1:])  # [step,stock_num] 从第1天开始
        # vol_before = np.divide(self.cash*y[0:-1, :], price_before[0:-1, :])
        # vol_now = np.divide(self.cash* y[1:, :], price_later[0:-1, :])

        rt = np.log(np.divide(np.sum(price_later *y_new , axis=1),
                              np.sum(price_before* y, axis=1)))

        self.returns = rt
        print('rt: ',rt)
        mean=np.mean(self.returns)
        var=np.var(self.returns)
        self.sharpe_ratio = mean / np.sqrt(var)

        index_before = np.transpose(self.index[0:-1])
        index_after = np.transpose(self.index[1:])
        # 超额收益
        self.ER = np.sum(np.mean(np.log(
            np.divide(np.sum(price_later*y_new,axis=1),np.sum( price_before*y,axis=1))) - np.log(np.divide(index_after, index_before))))
        # 跟踪误差
        self.TE = np.mean(np.sqrt(
            np.sum(np.square(
                np.log(np.divide(np.sum(price_later* y_new, axis=1),
                                 np.sum(price_before * y, axis=1))) -
                np.log(np.divide(index_after, index_before))))))

        self.IR=self.ER/self.TE

        # self.tracing_ratio = np.divide(
        #     np.divide(np.sum(self.all_market_capitalization, axis=0),
        #               np.sum(self.all_market_capitalization[:, 0], axis=0)),
        #     np.divide(np.sum(np.transpose(decisions) * self.all_market_capitalization, axis=0),
        #               np.sum(
        #                   np.transpose(decisions) * np.expand_dims(self.all_market_capitalization[:, 0], axis=1),
        #                   axis=0))
        # )

        return self.TE,self.ER,self.sharpe_ratio,self.IR

