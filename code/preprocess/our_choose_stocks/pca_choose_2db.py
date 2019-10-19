from preprocess.our_choose_stocks.pca import Pca
from preprocess.our_choose_stocks.dataKeeper import DataKeeper
from preprocess.our_choose_stocks.kmeans import KMEANS
import sqlite3
from preprocess.our_choose_stocks.read_config import *
from preprocess.our_choose_stocks.birch import BIRCH

class Pca_Choose_2db(object):
    def __init__(self):
        self.__datakeeper = DataKeeper()
        self.__pca=Pca(self.__datakeeper)
        self.__all_days_features=self.__pca.get_all_day_transform()
        self.__cluster_type=config.get('cluster','type')
        if self.__cluster_type=='kmeans':
            self.__cluster_model=KMEANS(self.__datakeeper,self.__pca.get_latest_day_transform())
        if self.__cluster_type=='birch':
            self.__cluster_model=BIRCH(self.__datakeeper,self.__pca.get_latest_day_transform())
        self.__choose_stocks_index = self.__cluster_model.choose_stocks_index()

        self.__all_stock_code=self.__datakeeper.all_stocks_code
        self.__all_date=self.__datakeeper.all_date

        self.__all_days=int(config.get('data','all_days_used'))
        self.__features_num=int(config.get('pca','n_component'))
        self.__preprocess_type='norm'

        self.__choosen_stock_num=int(config.get('data','stock_choose_num'))
        self.__table_name=self.__preprocess_type+'_pca'+'_'+self.__cluster_type+'_feature'+str(self.__features_num)+\
                          '_choosen'+str(self.__choosen_stock_num)
        self.db_path=config.get('db','db_path')
    def __clearOrCreate_table(self,conn,name):
        feature_sql=''
        for i in range(self.__features_num):
            feature_sql=feature_sql+'feature'+str(i)+' double,'
        feature_sql=feature_sql+'closeprice double'
        SQL = 'create table if not exists '+name+'(code string,date date,'+feature_sql+')'
        conn.execute(SQL)
        SQL='delete from '+self.__table_name
        conn.execute(SQL)
        conn.execute(SQL)
        print('table '+self.__table_name+' create ready!')

    def data2db(self):
        conn = sqlite3.connect(self.db_path)
        print('choosen stocks index is:', str(self.__choose_stocks_index))
        choosen_code=self.__datakeeper.all_stocks_code[self.__choose_stocks_index]
        for code in choosen_code:
            print(code)
            SQL='insert into %s select * from originData where code=\"%s\"' % ('choosen_stocks_origin',code)
            conn.execute(SQL)
        print('choosen_stocks_origin ready!')
        self.__clearOrCreate_table(conn,self.__table_name)
        for index in self.__choose_stocks_index:
            closeprice=[]
            stock_code=self.__all_stock_code[index]
            print(stock_code)
            SQL = 'select closeprice from originData where code=\"%s\"' % stock_code
            cursor = conn.execute(SQL)
            result = cursor.fetchall()
            for row in result:
                closeprice.append(row[0])
            for i,day in enumerate(self.__all_date):
                features=self.__all_days_features[index,i,:]
                SQL='insert into '+self.__table_name+' values(\"'+stock_code+'\",'+str(day)+','
                for feature in features:
                    SQL=SQL+str(feature)+','
                SQL=SQL+str(closeprice[i])
                SQL=SQL+')'
                conn.execute(SQL)
        conn.commit()
        conn.close()



if __name__ == '__main__':
    todb=Pca_Choose_2db()
    todb.data2db()