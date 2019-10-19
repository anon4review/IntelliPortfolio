import configparser
import os
import numpy as np
import sqlite3

def getConfig(section, key):
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0]+'/config.ini'
    config.read(path)
    return config.get(section, key)

class GetData(object):
    def __init__(self):
        self.choosen_stocks_num = int(getConfig('rrl','choosen_stocks_num'))
        self.all_days_used = int(getConfig('rrl','all_days_used'))

        self.train_num = int(getConfig('rrl', 'train_num'))
        self.test_num = int(getConfig('rrl', 'test_num'))
        self.window_len = int(getConfig('rrl', 'window_len'))
        self.stockNum = int(getConfig('data', 'stockNum'))

        self.Ftminus1 = np.array([1.0 for _ in range(self.choosen_stocks_num)])
        self.Ftminus1 = self.Ftminus1 / np.sum(self.Ftminus1)

        self.stock = []
        self.all_stock = []
        self.close_price = np.array([[0.0 for _ in range(self.all_days_used)] for _ in range(self.choosen_stocks_num)])
        self.index = np.array([0.0 for _ in range(self.all_days_used)])

        # 市场资本总值  总市值
        self.market_capitalization = np.array([[0.0 for  _ in range(self.all_days_used)] for _ in range(self.choosen_stocks_num)])
        self.all_market_capitalization = np.array([[0.0 for _ in range(self.all_days_used)] for _ in range(self.stockNum)])

        db_path=getConfig('db', 'db_path')
        table_name = getConfig('db', 'table_name')
        conn = sqlite3.connect(db_path)

        # get stock
        SQL = "select distinct code from %s " % table_name
        cursor = conn.execute(SQL)
        data = cursor.fetchall()
        for row in data:
            self.stock.append(row[0])
        print(self.stock)

        # get all stock
        SQL = "select distinct code from originData"
        cursor = conn.execute(SQL)
        data = cursor.fetchall()
        for row in data:
            self.all_stock.append(row[0])

        # get close_price
        stock_i = 0
        for stock in self.stock:
            SQL = "select closeprice from originData where code = \'%s\'" % stock
            cursor = conn.execute(SQL)
            data = cursor.fetchall()
            date_i = 0
            for row in data:
                self.close_price[stock_i][date_i] = row[0]
                date_i += 1
            stock_i += 1

        # # get choosen stock TotalValue
        # stock_i = 0
        # for stock in self.stock:
        #     SQL = 'select TotalValue from originData where code=\'%s\'' % stock
        #     cursor = conn.execute(SQL)
        #     data = cursor.fetchall()
        #     date_i = 0
        #     for row in data:
        #         self.market_capitalization[stock_i][date_i] = row[0]
        #         date_i += 1
        #     stock_i += 1
        #
        # # get all stock TotalValue
        # stock_i = 0
        # for stock in self.all_stock:
        #     SQL = 'select TotalValue from originData where code=\'%s\'' % stock
        #     cursor = conn.execute(SQL)
        #     data = cursor.fetchall()
        #     date_i = 0
        #     for row in data:
        #         self.all_market_capitalization[stock_i][date_i] = row[0]
        #         date_i += 1
        #     stock_i += 1

        # get index_point
        # SQL = "select index_point from stock_index where date>=%d and date <=%d order by date ASC"%(
        #     self.beginTime-self.timeInterval, self.endTime
        # )
        SQL = "select closeprice from indexes"
        cursor = conn.execute(SQL)
        data = cursor.fetchall()
        date_i = 0
        for row in data:
            self.index[date_i] = row[0]
            date_i += 1


        # calculate rt
        # self.rt = self.close_price[:, 1:] - self.close_price[:, 0:self.all_days_used]
        # # normalization
        # min_volume = np.min(self.rt, axis=1, keepdims=True)
        # max_volume = np.max(self.rt, axis=1, keepdims=True)
        # self.rt = np.divide((self.rt - min_volume), (max_volume - min_volume))
        self.rt = np.log(self.close_price[:, 1:] / self.close_price[:, 0:self.all_days_used - 1])
        self.rt = np.concatenate([np.ones((self.choosen_stocks_num, 1), dtype=float), self.rt], axis=1)

        # 归一化
        # minrt = np.min(self.rt, keepdims=True, axis=1)
        # maxrt = np.max(self.rt, keepdims=True, axis=1)
        # self.rt = np.divide((self.rt - minrt), (maxrt - minrt))

        # self.train_rt = self.rt[:, 0:self.train_num + self.window_len-1]
        self.train_rt = self.rt[:, 0:self.train_num]
        # meanrt_train = np.mean(self.train_rt)
        # varrt_train = np.var(self.train_rt)
        # self.train_rt = (self.train_rt - meanrt_train) / np.sqrt(varrt_train)
        # 原始数据计算指标
        # self.train_price = self.close_price[:,self.window_len-2:self.train_num + self.window_len-1]
        self.train_price = self.close_price[:, self.window_len - 1:self.train_num]
        # self.train_index = self.index[self.window_len - 1:self.train_num + self.window_len - 1]
        self.train_index = self.index[self.window_len-1:self.train_num]
        # self.train_market_capitalization = self.market_capitalization[:,
        #                                    self.window_len - 2:self.train_num + self.window_len - 1]
        self.train_market_capitalization = self.market_capitalization[:,
                                           self.window_len - 1:self.train_num]
        # self.train_all_market_capitalization = self.all_market_capitalization[:,
        #                                        self.window_len - 2:self.train_num + self.window_len - 1]
        self.train_all_market_capitalization = self.all_market_capitalization[:,
                                               self.window_len - 1:self.train_num]

        # self.eva_rt = self.rt[:, self.train_num:self.train_num + self.eva_num + self.window_len-1]
        # self.eva_price = self.close_price[:, self.window_len-2+self.train_num:self.train_num + self.eva_num + self.window_len-1]

        # self.test_rt = self.rt[:,
        #                 self.train_num:self.train_num + self.test_num + self.window_len - 1]
        self.test_rt = self.rt[:,
                       self.train_num:self.train_num + self.test_num]
        # self.test_rt = (self.test_rt - meanrt_train) / np.sqrt(varrt_train)
        # 原始数据计算指标
        # self.test_price = self.close_price[:,
        #                    self.window_len - 2 + self.train_num:self.train_num + self.test_num + self.window_len - 1]
        self.test_price = self.close_price[:,
                          self.window_len - 1 + self.train_num:self.train_num + self.test_num]

        # self.test_index = self.index[self.window_len-1 + self.train_num : self.train_num +self.test_num +self.window_len -1]
        self.test_index = self.index[self.window_len - 1 + self.train_num: self.train_num + self.test_num]
        # self.test_market_capitalization = self.market_capitalization[:,
        #                                   self.window_len - 2 + self.train_num :self.train_num + self.test_num + self.window_len - 1]
        self.test_market_capitalization = self.market_capitalization[:,
                                          self.window_len - 1 + self.train_num:self.train_num + self.test_num]
        # self.test_all_market_capitalization = self.all_market_capitalization[:,
        #                                       self.window_len - 2 + self.train_num :self.train_num + self.test_num + self.window_len - 1]
        self.test_all_market_capitalization = self.all_market_capitalization[:,
                                              self.window_len - 1 + self.train_num:self.train_num + self.test_num]

    def train1(self):
        train_rt = self.train_rt[:, 0:self.window_len]
        train_rt = self.get_Standardization_rt(train_rt)
        train_rt = np.reshape(train_rt, [self.choosen_stocks_num, self.window_len, 1])
        final_data = train_rt.reshape((1,self.choosen_stocks_num*self.window_len))
        # for i in range(self.train_num - 1):
        for i in range(self.train_num - self.window_len):
            train_rt_temp = self.train_rt[:, i+1:self.window_len+i+1]
            train_rt_temp = self.get_Standardization_rt(train_rt_temp)
            # train_rt_temp = np.reshape(train_rt_temp, [self.choosen_stocks_num, self.window_len, 1])
            # train_rt_temp = np.reshape(train_rt_temp, [self.choosen_stocks_num,1, 1])
            concate = train_rt_temp.reshape((1, self.choosen_stocks_num*self.window_len))
            final_data = np.concatenate((final_data, concate), axis=0)
        # final_data = final_data.reshape((1, self.train_num, self.choosen_stocks_num*self.window_len))
        final_data = final_data.reshape((1, self.train_num - self.window_len + 1, self.choosen_stocks_num * self.window_len))
        return final_data

    def test(self):
        test_rt = self.test_rt[:, 0:self.window_len]
        test_rt = self.get_Standardization_rt(test_rt)
        test_rt = np.reshape(test_rt, [self.choosen_stocks_num, self.window_len, 1])
        final_data = test_rt.reshape((1,self.choosen_stocks_num*self.window_len))
        for i in range(self.test_num - self.window_len):
            test_rt_temp = self.test_rt[:, i+1:self.window_len+i+1]
            test_rt_temp = self.get_Standardization_rt(test_rt_temp)
            # test_rt_temp = np.reshape(test_rt_temp, [self.choosen_stocks_num, self.window_len, 1])
            concate = test_rt_temp.reshape((1, self.choosen_stocks_num*self.window_len))
            final_data = np.concatenate((final_data, concate), axis=0)
        final_data = final_data.reshape((1, self.test_num - self.window_len +1, self.choosen_stocks_num*self.window_len))
        return final_data

    def get_Standardization_rt(self, data):
        meanrt_data = np.mean(data)
        varrt_data = np.var(data)
        data = (data - meanrt_data) / np.sqrt(varrt_data)
        return data

if __name__=='__main__':
    getData = GetData()
    print(getData.train1())