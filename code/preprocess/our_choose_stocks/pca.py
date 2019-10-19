from sklearn.decomposition import PCA
from preprocess.our_choose_stocks.read_config import *
import numpy as np

class Pca(object):
    def __init__(self,datakeep):
        self.__dataKeep=datakeep
        self.__n_component = int(config.get('pca', 'n_component'))
        self.__pca = PCA(n_components=self.__n_component)
        self.__stock_num = int(config.get('data', 'stock_num'))
        self.__latest_days_features=self.__dataKeep.latest_days_features
        self.__latest_days_num=self.__dataKeep.latest_days_num
        self.__all_days_features=self.__dataKeep.all_days_features
        self.__all_days_num=int(config.get('data','all_days_used'))
        self.__train()
        print('PCA completed')

    def __train(self):
        self.__pca.fit(self.__latest_days_features)

    def get_latest_day_transform(self):
        new_latest_features = self.__pca.transform(self.__latest_days_features)
        #[latest_day*111,3] to [latest_day,111,3]
        slice_index=np.array([i for i in range(self.__stock_num)])
        new_latest_features=np.array([new_latest_features[slice_index*self.__latest_days_num+i] for i in range(self.__latest_days_num)])
        return new_latest_features

    def get_all_day_transform(self):
        new_all_features=self.__pca.transform(self.__all_days_features)
        #[all_days*111,3] to [111,all_day,3]
        print(new_all_features.shape)
        new_all_features=new_all_features.reshape(self.__stock_num,self.__all_days_num,self.__n_component)
        return new_all_features




if __name__ == '__main__':
    pca=Pca()
    pca.get_latest_day_transform()
