from sklearn.cluster import Birch # 层次聚类
import numpy as np
from preprocess.our_choose_stocks.read_config import *

class BIRCH(object):
    def __init__(self,datakeep,X):
        self.__datakeep = datakeep
        self.__stock_num = int(config.get('data','stock_num'))
        self.__X = X
        self.__n_clusters = int(config.get('data','stock_choose_num'))

    def choose_stocks_index(self):
        stock_choosen_num = {}
        for i in range(self.__X.shape[0]):
            birch = Birch(threshold=0.001,n_clusters=self.__n_clusters)
            y_pred = birch.fit_predict(self.__X[i, :, :])
            subcluster_centers = birch.subcluster_centers_

            choosen_stock = np.array([0 for _ in range(self.__n_clusters)])
            min_distance = np.array([-1.0 for _ in range(self.__n_clusters)])
            for ind in range(self.__stock_num):
                stock = self.__X[i, ind, :]
                stock_label = y_pred[ind]
                distance = np.linalg.norm(stock - subcluster_centers[stock_label], ord=2)
                if min_distance[stock_label] == -1 or min_distance[stock_label] > distance:
                    min_distance[stock_label] = distance
                    choosen_stock[stock_label] = ind
            for stock in choosen_stock:
                if stock in stock_choosen_num.keys():
                    stock_choosen_num[stock] += 1
                else:
                    stock_choosen_num[stock] = 1
        stock_choosen_num = list(sorted(stock_choosen_num.items(), key=lambda x: x[1], reverse=True))
        choosen_stock_ind = list(map(lambda x: x[0], stock_choosen_num))[:self.__n_clusters]
        return choosen_stock_ind

if __name__=='__main__':
    pass