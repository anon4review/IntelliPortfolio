import sqlite3
import numpy as np
from configparser import ConfigParser

'''将相关数据从数据库中读取出来、存储，以节省访问时间'''

class DataKeeper(object):
    def __init__(self):
        cp = ConfigParser()
        cp.read('config.conf')
        self.N = int(cp.get('hy', 'N'))
        self.all_days_used = int(cp.get('hy', 'all_days_used'))
        self.code = [0 for i in range(self.N)]
        self.data=np.array([[0.0 for i in range(self.all_days_used)] for j in range(self.N)] ) #K支股票的价格数据 k*(T+L)
        self.index = np.array([0.0 for i in range(self.all_days_used)])# 指数的价格数据 T+L
        self.db_path=cp.get('db', 'db_path')
        # 市场资本总值   总市值
        self.all_market_capitalization = np.array(
            [[0.0 for _ in range(self.all_days_used)] for _ in range(self.N)])
        self.memory_stocks_data()

    def memory_stocks_data(self):
        '''获得N支股票的相关信息并存储'''
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        #print('数据库连接成功')

        #获得N支股票code
        sql="select distinct code from originData"
        c.execute(sql)
        for i in range(self.N):
            result=c.fetchone()
            self.code[i]=result[0]


        # get stock price
        stock_i = 0
        for code in self.code:
            SQL = 'select closeprice from originData where code=\'%s\'' % code
            cursor = conn.execute(SQL)
            data = cursor.fetchall()
            date_i = 0
            for row in data:
                self.data[stock_i][date_i] = row[0]
                date_i += 1
            stock_i += 1

        # # get all stock TotalValue
        # stock_i = 0
        # for code in self.code:
        #     SQL = 'select TotalValue from originData where code=\'%s\'' % code
        #     cursor = conn.execute(SQL)
        #     data = cursor.fetchall()
        #     date_i = 0
        #     for row in data:
        #         self.all_market_capitalization[stock_i][date_i] = row[0]
        #         date_i += 1
        #     stock_i += 1

        # get index
        SQL = 'select closeprice from indexes'
        cursor = conn.execute(SQL)
        data = cursor.fetchall()
        date_i = 0
        for row in data:
            self.index[date_i] = row[0]
            date_i += 1

        conn.commit()
        conn.close()

if __name__=="__main__":
    datak=DataKeeper()