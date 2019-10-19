import sklearn.cluster
from preprocess.our_choose_stocks.read_config import *
import numpy as np

class KMEANS(object):
    def __init__(self,datakeep,X):
        self.__datakeep=datakeep
        self.__n_clusters = int(config.get('data','stock_choose_num'))
        self.__stock_num = int(config.get('data','stock_num'))
        self.__X = X

    def choose_stocks_index(self):
        stock_choosen_num={}
        for i in range(self.__X.shape[0]):
            centroid, label, _, _ = sklearn.cluster.k_means(self.__X[i,:,:], n_clusters=self.__n_clusters,
                                                                      return_n_iter=True)
            choosen_stock = np.array([0 for _ in range(self.__n_clusters)])
            min_distance = np.array([-1.0 for _ in range(self.__n_clusters)])
            for ind in range(self.__stock_num):
                stock = self.__X[i, ind, :]
                stock_label = label[ind]
                distance = np.linalg.norm(stock - centroid[stock_label], ord=2)
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

if __name__ == '__main__':
    pass


