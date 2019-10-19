import configparser
import os
import numpy as np
import sqlite3

def getConfig(section, key):
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + '/config.ini'
    config.read(path)
    return config.get(section, key)

class DataGenerator(object):
    def __init__(self):
        self.train_num=int(getConfig('train','train_num'))
        self.trade_num = int(getConfig('trade', 'trade_num'))
        self.eva_num = int(getConfig('eva', 'eva_num'))
        self.test_num = int(getConfig('test', 'test_num'))
        self.indicator_num=int(getConfig('rrl', 'indicator_num'))
        self.all_days_used=int(getConfig('rrl', 'all_days_used'))
        self.choosen_stocks_num=int(getConfig('rrl', 'choosen_stocks_num'))
        self.window_len = int(getConfig('rrl', 'window_len'))
        # self.stockNum=int(getConfig('data', 'stockNum'))
        # self.choose_begin_date=int(getConfig('choose','begin_date'))
        # self.choose_end_date=int(getConfig('choose','end_date'))
        Ftminus1 = np.array([1.0 for _ in range(self.choosen_stocks_num)])
        self.Ftminus1 = Ftminus1 / np.sum(Ftminus1)
        self.db_path = getConfig('db', 'db_path')
        self.table_name = getConfig('db', 'table_name')
        self.index_table = getConfig('db', 'index_table_name')
        self.code=[]
        self.indicator_data=np.array([[[0.0 for _ in range(self.indicator_num)] for _ in range(self.all_days_used)] for _ in range(self.choosen_stocks_num)])
        self.price=np.array([[0.0 for _ in range(self.all_days_used)] for _ in range(self.choosen_stocks_num)])
        self.index = np.array([0.0 for _ in range(self.all_days_used)] )
        conn = sqlite3.connect(self.db_path)

        # get stock code
        # SQL = "select code,sum(vol) from %s where date >=%d and date<=%d group by code order by sum(vol) DESC limit %d" % (
        # self.table_name,self.choose_begin_date, self.choose_end_date, self.choosen_stocks_num)
        SQL = "select distinct code from %s" % self.table_name
        cursor = conn.execute(SQL)
        data = cursor.fetchall()
        for row in data:
            self.code.append(row[0])
        print('choosen stocks:',self.code)

        # get indicator
        stock_i = 0
        for code in self.code:
            SQL='select * from %s where code=\'%s\'' % (self.table_name,code)
            cursor=conn.execute(SQL)
            data=cursor.fetchall()
            date_i=0
            for row in data:
                for ind_i in range(self.indicator_num):
                    self.indicator_data[stock_i][date_i][ind_i]=row[ind_i+2]  ########!!!!!!!!!!!!!!!!!!!!!!!
                date_i+=1
            stock_i+=1

        # get stock price
        stock_i = 0
        for code in self.code:
            SQL = 'select closeprice from %s where code=\'%s\'' % (self.table_name,code)
            cursor = conn.execute(SQL)
            data = cursor.fetchall()
            date_i = 0
            for row in data:
                self.price[stock_i][date_i] = row[0]
                date_i += 1
            stock_i += 1

        # get index
        SQL = 'select closeprice from %s '% self.index_table
        cursor = conn.execute(SQL)
        data = cursor.fetchall()
        date_i = 0
        for row in data:
            self.index[date_i] = row[0]
            date_i += 1

        #train
        self.train_indicator_data = self.indicator_data[:, 0:self.train_num , :]
        self.train_price = self.price[:, self.window_len-1:self.train_num]
        self.train_index = self.index[self.window_len-1:self.train_num]
        #eva
        self.eva_indicator_data = self.indicator_data[:, self.train_num:self.train_num + self.eva_num ,:]
        self.eva_price = self.price[:, self.train_num + self.window_len - 1:self.train_num + self.eva_num]
        self.eva_index = self.index[self.train_num + self.window_len - 1:self.train_num + self.eva_num]
        # trade
        self.trade_indicator_data = self.indicator_data[:,self.train_num + self.eva_num:self.train_num + self.trade_num + self.eva_num , :]
        self.trade_price = self.price[:, self.train_num + self.eva_num+self.window_len-1:self.train_num + self.trade_num + self.eva_num]
        self.trade_index = self.index[self.train_num + self.eva_num + self.window_len - 1:self.train_num + self.trade_num + self.eva_num]
        #test
        self.test_indicator_data = self.indicator_data[:,
                                    self.train_num + self.eva_num+self.trade_num:self.train_num + self.trade_num + self.eva_num +self.test_num,
                                    :]
        self.test_price = self.price[:,
                           self.train_num + self.eva_num+self.trade_num+self.window_len-1:self.train_num + self.trade_num + self.eva_num +self.test_num]
        self.test_index=self.index[self.window_len-1+self.train_num + self.eva_num+self.trade_num:self.train_num + self.trade_num + self.eva_num +self.test_num]

    def get_train_data(self,ga):
        temp = np.tile(ga[:self.indicator_num], (self.choosen_stocks_num, self.window_len, 1))
        train_data=self.train_indicator_data[:,0:self.window_len,:]
        train_data=self.get_Standardization_data(train_data)
        train_data=train_data*temp
        final_data = train_data.reshape((1, self.choosen_stocks_num * self.window_len * self.indicator_num))
        for i in range(self.train_num-self.window_len):
            train_data=self.train_indicator_data[:, i+1:self.window_len+i+1, :]
            train_data=self.get_Standardization_data(train_data)
            train_data = train_data*temp
            train_data = train_data.reshape((1, self.choosen_stocks_num * self.window_len * self.indicator_num))
            final_data=np.concatenate((final_data,train_data),axis=0)
        final_data=final_data.reshape((1,self.train_num-self.window_len+1,self.choosen_stocks_num * self.window_len * self.indicator_num))
        return final_data

    def get_eva_data(self, ga):
        temp = np.tile(ga[:self.indicator_num], (self.choosen_stocks_num, self.window_len, 1))
        eva_data = self.eva_indicator_data[:, 0:self.window_len, :]
        eva_data = self.get_Standardization_data(eva_data)
        eva_data = eva_data * temp
        final_data = eva_data.reshape((1, self.choosen_stocks_num * self.window_len * self.indicator_num))
        for i in range(self.eva_num - self.window_len):
            eva_data = self.eva_indicator_data[:, i + 1:self.window_len + i + 1, :]
            eva_data = self.get_Standardization_data(eva_data)
            eva_data = eva_data * temp
            eva_data = eva_data.reshape((1, self.choosen_stocks_num * self.window_len * self.indicator_num))
            final_data = np.concatenate((final_data, eva_data), axis=0)
        final_data = final_data.reshape((1, self.eva_num-self.window_len+1, self.choosen_stocks_num * self.window_len * self.indicator_num))
        return final_data

    def get_trade_data(self, ga):
        temp = np.tile(ga[:self.indicator_num], (self.choosen_stocks_num, self.window_len, 1))
        trade_data = self.trade_indicator_data[:, 0:self.window_len, :]
        trade_data = self.get_Standardization_data(trade_data)
        trade_data = trade_data * temp
        final_data = trade_data.reshape((1, self.choosen_stocks_num * self.window_len * self.indicator_num))
        for i in range(self.trade_num - self.window_len):
            trade_data = self.trade_indicator_data[:, i + 1:self.window_len + i + 1, :]
            trade_data = self.get_Standardization_data(trade_data)
            trade_data*=temp
            trade_data = trade_data.reshape((1, self.choosen_stocks_num * self.window_len * self.indicator_num))
            final_data = np.concatenate((final_data, trade_data), axis=0)
        final_data = final_data.reshape((1, self.trade_num-self.window_len+1, self.choosen_stocks_num * self.window_len * self.indicator_num))
        return final_data

    def get_test_data(self, ga):
        temp = np.tile(ga[:self.indicator_num], (self.choosen_stocks_num, self.window_len, 1))
        test_data = self.test_indicator_data[:, 0:self.window_len, :]
        test_data = self.get_Standardization_data(test_data)
        test_data = test_data * temp
        final_data = test_data.reshape((1, self.choosen_stocks_num * self.window_len * self.indicator_num))
        for i in range(self.test_num-self.window_len):
            test_data = self.test_indicator_data[:, i + 1:self.window_len + i + 1, :]
            test_data = self.get_Standardization_data(test_data) * temp
            test_data = test_data.reshape((1, self.choosen_stocks_num * self.window_len * self.indicator_num))
            final_data = np.concatenate((final_data, test_data), axis=0)
        final_data = final_data.reshape((1, self.test_num-self.window_len+1, self.choosen_stocks_num * self.window_len * self.indicator_num))
        return final_data

    def get_Standardization_data(self,data):
        data = data.reshape(-1, self.indicator_num)
        mean_data = np.mean(data, axis=0)
        var_data = np.var(data, axis=0)
        data = data.reshape(self.choosen_stocks_num, -1, self.indicator_num)
        data = (data - mean_data) / np.sqrt(var_data)
        return data

if __name__=='__main__':
    data=DataGenerator()