from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
from pgportfolio.tools.data import panel_fillna
from pgportfolio.constants import *
import sqlite3
from datetime import datetime
import logging


class HistoryManager:
    # if offline ,the coin_list could be None
    # NOTE: return of the sqlite results is a list of tuples, each tuple is a row
    def __init__(self,coin_number, online=True):
        self.__storage_period = DAY  # keep this as 86400
        self._coin_number = coin_number
        self._online = online
        self.__coins = None

    @property
    def coins(self):
        return self.__coins

    def get_global_data_matrix(self, features=('close',)):
        """
        :return a numpy ndarray whose axis is [feature, coin, time]
        """
        return self.get_global_panel(features).values

    def get_stockIndex_panel(self):
        date_list = self.return_date()
        #time_index = pd.to_datetime(list(date_list), unit='D')
        panel = pd.Panel(items=[1],major_axis=[1],minor_axis=date_list, dtype=np.float32)
        connection = sqlite3.connect(DATABASE_DIR)
        print('######')
        print(panel)
        try:
            sql = ("SELECT date , closeprice FROM indexes")
            data=connection.execute(sql).fetchall()
            for da in data:
                panel.loc[1,1,int(da[0])] = da[1]
            print(panel)

            #panel = panel_fillna(panel, "both")
        finally:
            connection.commit()
            connection.close()

        return panel

    def get_global_panel(self, features=('close',)):#, period=86400
        ''':param start/end: linux timestamp in seconds
        :param period: time interval of each data access point
        :param features: tuple or list of the feature names
        :return a panel, [feature, coin, time]
        '''
        stocks = self.select_coins()
        self.__coins = stocks
        self.stock_code = stocks
        # print(self.__coins)
        # print(period)
        date_list=self.return_date()

        if len(stocks)!=self._coin_number:
            raise ValueError("the length of selected coins %d is not equal to expected %d"
                             % (len(stocks), self._coin_number))

        logging.info("feature type list is %s" % str(features))
        #self.__checkperiod(period)

        # time_index = pd.to_datetime(list(range(start, end+1, 3600)),unit='s')
        # print(time_index)
        panel = pd.Panel(items=features, major_axis=stocks, minor_axis=date_list, dtype=np.float32)
        #print(panel.shape)

        connection = sqlite3.connect(DATABASE_DIR)
        try:
            for row_number, code in enumerate(stocks):
                for feature in features:
                    if feature == "close":
                        sql = 'SELECT date, closeprice FROM originData WHERE code= \"%s\" '%code
                    elif feature == "high":
                        sql = 'SELECT date, maxprice FROM originData WHERE code= \"%s\" '%code
                    elif feature == "low":
                        sql = 'SELECT date, minprice FROM originData WHERE code= \"%s\" '%code

                    else:
                        msg = ("The feature %s is not supported" % feature)
                        logging.error(msg)
                        raise ValueError(msg)
                    data=connection.execute(sql).fetchall()
                    for da in data:
                        panel.loc[feature, code,int(da[0])] = da[1]

        finally:
            connection.commit()
            connection.close()
        return panel

    # def get_market_capticalization(self):
    #     ''':param start/end: linux timestamp in seconds
    #     :param period: time interval of each data access point
    #     :param features: tuple or list of the feature names
    #     :return a panel, [feature, coin, time]
    #     '''
    #     stocks = self.select_coins()
    #     self.__coins = stocks
    #
    #     date_list=self.return_date()
    #
    #     if len(stocks)!=self._coin_number:
    #         raise ValueError("the length of selected coins %d is not equal to expected %d"
    #                          % (len(stocks), self._coin_number))
    #
    #     panel = pd.Panel(items=["market_capticalization"], major_axis=stocks, minor_axis=date_list, dtype=np.float32)
    #
    #     connection = sqlite3.connect(DATABASE_DIR)
    #     try:
    #         for row_number, code in enumerate(stocks):
    #
    #             sql = 'SELECT date, TotalValue FROM originData WHERE code= \"%s\" '%code
    #
    #             data=connection.execute(sql).fetchall()
    #             for da in data:
    #                 panel.loc["market_capticalization", code,int(da[0])] = da[1]
    #
    #     finally:
    #         connection.commit()
    #         connection.close()
    #     return panel
    #
    # def get_all_market_capticalization(self):
    #     ''':param start/end: linux timestamp in seconds
    #     :param period: time interval of each data access point
    #     :param features: tuple or list of the feature names
    #     :return a panel, [feature, coin, time]
    #     '''
    #     stocks = self.select_all_stocks()
    #
    #     date_list=self.return_date()
    #
    #
    #     panel = pd.Panel(items=["all_market_capticalization"], major_axis=stocks, minor_axis=date_list, dtype=np.float32)
    #
    #     connection = sqlite3.connect(DATABASE_DIR)
    #     try:
    #         for row_number, code in enumerate(stocks):
    #
    #             sql = 'SELECT date, TotalValue FROM originData WHERE code= \"%s\" '%code
    #
    #             data=connection.execute(sql).fetchall()
    #             for da in data:
    #                 panel.loc["all_market_capticalization", code,int(da[0])] = da[1]
    #
    #     finally:
    #         connection.commit()
    #         connection.close()
    #     return panel

    # select top coin_number of coins by volume from start to end
    def select_coins(self):
        SQL='select distinct code from %s' % TABLE_NAME
        connection = sqlite3.connect(DATABASE_DIR)
        cursor=connection.execute(SQL)
        coins_tuples = cursor.fetchall()
        if len(coins_tuples) != self._coin_number:
            logging.error("the sqlite error happend")
        connection.close()
        coins = []
        for tuple in coins_tuples:
            coins.append(tuple[0])

        return coins

    def select_all_stocks(self):

        connection = sqlite3.connect(DATABASE_DIR)
        try:
            cursor=connection.cursor()
            cursor.execute('SELECT distinct code FROM originData')
            coins_tuples = cursor.fetchall()

        finally:
            connection.commit()
            connection.close()
        coins = []
        for tuple in coins_tuples:
            coins.append(tuple[0])

        logging.debug("All coins are: "+str(coins))
        return coins

    # add new history data into the database
    def return_date(self):
        connection = sqlite3.connect(DATABASE_DIR)
        cursor = connection.cursor()
        date = cursor.execute('SELECT distinct date FROM originData ').fetchall()
        date_result=[int(item[0]) for item in date ]
        connection.close()
        return date_result

