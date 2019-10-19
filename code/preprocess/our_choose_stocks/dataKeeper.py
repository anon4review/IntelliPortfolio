import sqlite3
from preprocess.our_choose_stocks.read_config import *
import numpy as np

class DataKeeper(object):
    def __init__(self):
        self.__start_date=int(config.get('data','start_date'))
        self.__end_date=int(config.get('data','end_date'))
        db_path=config.get('db','db_path')
        self.__conn = sqlite3.connect(db_path)
        self.__stock_num = int(config.get('data', 'stock_num'))
        self.all_stocks_code = self.__get_all_stock_code()
        self.all_date = self.__get_all_date()
        self.latest_days_features = self.__get_latest_features()
        self.all_days_features = self.__get_all_features()
        self.latest_days_num=int(self.latest_days_features.shape[0]/self.__stock_num)
        self.__conn.close()

    def __get_latest_features(self):
        latest_features=[]
        SQL='select * from nor_data where date>=%d and date<=%d' %(self.__start_date,self.__end_date)
        result=self.__conn.execute(SQL).fetchall()
        for row in result:
            temp_data=list(row)[3:]
            latest_features.append(temp_data)
        latest_features=np.array(latest_features)
        return latest_features

    def __get_all_features(self):
        all_features=[]
        SQL='select * from nor_data'
        result = self.__conn.execute(SQL).fetchall()
        for row in result:
            temp_data = list(row)[3:]
            all_features.append(temp_data)
        all_features = np.array(all_features)
        return all_features


    def __get_all_stock_code(self):
        code=[]
        SQL='select distinct code from originData'
        result = self.__conn.execute(SQL).fetchall()
        for row in result:
            code.append(row[0])
        code=np.array(code)
        return code

    def __get_all_date(self):
        date=[]
        SQL='select distinct date from originData'
        result = self.__conn.execute(SQL).fetchall()
        for row in result:
            date.append(row[0])
        date=np.array(date)
        return date


