import tensorflow as tf
import numpy as np
import configparser
import os
import tqdm
import sqlite3
from preprocess.norm_6_features import norm

def getConfig(section, key):
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + '/config.ini'
    config.read(path)
    return config.get(section, key)

db_path=getConfig('db', 'db_path')
conn = sqlite3.connect(database=db_path)

class DataGenerator(object):
    def __init__(self):
        self.train_num=int(getConfig('autoencoder','train_num'))
        self.test_num = int(getConfig('autoencoder', 'test_num'))
        self.stock_num=int(getConfig('autoencoder', 'stockNum'))
        self.window_len=int(getConfig('autoencoder', 'window_len'))
        self.all_days_used=int(getConfig('autoencoder', 'all_days_used'))
        #stock code
        self.code=[]
        #stock price
        self.price=np.array([[0.0 for _ in range(self.all_days_used)] for _ in range(self.stock_num)])

        #get stock code
        SQL = "select distinct code from originData"
        cursor = conn.execute(SQL)
        for row in cursor:
            self.code.append(row[0])

        # get stock price
        stock_i = 0
        for code in self.code:
            SQL='select closeprice from originData where code=\'%s\'' %code
            cursor=conn.execute(SQL)
            date_i=0
            for row in cursor:
                self.price[stock_i][date_i]=row[0]
                date_i+=1
            stock_i+=1

        # standardized processing
        mean=np.mean(self.price,axis=0)
        var=np.var(self.price,axis=0)
        self.standard_price = (self.price-mean)/np.sqrt(var)

class AutoEncoder(object):
    def __init__(self):
        self.train_num = int(getConfig('autoencoder', 'train_num'))
        self.test_num = int(getConfig('autoencoder', 'test_num'))
        self.stock_num = int(getConfig('autoencoder', 'stockNum'))
        self.window_len = int(getConfig('autoencoder', 'window_len'))
        self.all_days_used = int(getConfig('autoencoder', 'all_days_used'))
        self.hidden_units = int(getConfig('autoencoder', 'hidden_units'))
        self.code_units = int(getConfig('autoencoder', 'code_units'))
        self.learning_rate = float(getConfig('autoencoder', 'learning_rate'))
        self.train_epoch = int(getConfig('autoencoder', 'train_epoch'))
        self.choosen_stocks_num = int(getConfig('autoencoder', 'choosen_stocks_num'))
        self.data=DataGenerator()
        self.sess=tf.Session()
        self.build()

    def build(self):
        #输入是1个batch的收盘价数据
        self.stocks_win_price=tf.placeholder(dtype=float,shape=[1,self.stock_num*self.window_len])

        hiddenlayer1=tf.layers.dense(self.stocks_win_price,units=self.hidden_units,activation=tf.nn.relu,
                                     name='hiddenlayer1')
        code=tf.layers.dense(hiddenlayer1,self.code_units,activation=tf.nn.relu,
                             name='code')
        hiddenlayer2=tf.layers.dense(code,units=self.hidden_units,activation=tf.nn.relu,
                                     name='hiddenlayer2')
        output=tf.layers.dense(hiddenlayer2,units=self.stock_num*self.window_len,activation=None,
                               name='output')
        self.output=output

        regularizer=tf.contrib.layers.l2_regularizer(0.1, scope=None)
        tf.contrib.layers.apply_regularization(regularizer, weights_list=tf.get_collection(tf.GraphKeys.VARIABLES))
        self.regularization_loss=tf.losses.get_regularization_loss()
        self.loss=tf.losses.mean_squared_error(labels=self.stocks_win_price,predictions=output)+\
                  self.regularization_loss
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step=self.optimizer.minimize(self.loss)

        self.loss_list=[]
        for i in range(self.stock_num):
            loss=tf.losses.mean_squared_error(labels=
                                              self.stocks_win_price[:,i*self.window_len:(i+1)*self.window_len],
                                              predictions=output[:,i*self.window_len:(i+1)*self.window_len])
            self.loss_list.append(loss)

    def train(self):
        standard_price=self.data.standard_price
        train_standard_price=standard_price[:,0:self.train_num]
        loop_time=self.train_num-self.window_len+1

        self.sess.run(tf.initialize_all_variables())
        for epoch in range(self.train_epoch):
            for loop in tqdm.tqdm(range(loop_time)):
                stocks_win_price=train_standard_price[:,loop:loop+self.window_len]
                stocks_win_price=np.reshape(stocks_win_price,[1,self.stock_num*self.window_len])
                feed={
                    self.stocks_win_price:stocks_win_price
                }
                _,loss,output,re_loss=self.sess.run([self.train_step,self.loss,self.output,self.regularization_loss],feed_dict=feed)
                print('loss:',loss)
                print('re_loss:',re_loss)

    def choose_stocks(self):
        standard_price = self.data.standard_price
        test_standard_price = standard_price[:, self.train_num:self.train_num+self.test_num]
        stocks_win_price = test_standard_price[:, 0:self.window_len]
        stocks_win_price = np.reshape(stocks_win_price, [1, self.stock_num * self.window_len])
        feed={
            self.stocks_win_price: stocks_win_price
        }
        self.train()
        loss_list=self.sess.run(self.loss_list,feed_dict=feed)

        #min loss:max loss=3:5
        min_loss_num=int(self.choosen_stocks_num*3/8)
        max_loss_num=int(self.choosen_stocks_num-min_loss_num)

        index_list=[i for i in range(self.stock_num)]
        loss_dict=dict(zip(index_list,loss_list))
        loss_list_pair=sorted(loss_dict.items(),key=lambda x:x[1])

        #print(loss_list_pair)
        choosen_stocks_index=[]
        for i in range(min_loss_num):
            choosen_stocks_index.append(loss_list_pair[i][0])
        for i in range(max_loss_num):
            choosen_stocks_index.append(loss_list_pair[-1-i][0])

        return np.array(choosen_stocks_index)

    def __clearOrCreate_table(self,conn,name):

        SQL = 'create table if not exists %s (code string,date date,openprice double,' \
              'maxprice double,minprice double, closeprice double,vol double,money double,updown double,' \
              'updownratio double,meanprice double, TurnoverRate double,CirculationValue double,TotalValue double,' \
              'FlowEquity double,TotalEquity double,PE double,PB double,P2S double,PCF double)'%name

        # SQL = 'create table if not exists %s (code string,date date,openprice double,' \
        #       'maxprice double,minprice double, closeprice double,vol double,adjcloseprice double)' % name

        conn.execute(SQL)
        SQL='delete from %s'%name
        conn.execute(SQL)
        print('table '+str(name)+' create ready!')


    def store_to_db(self,conn,index,table_name):
        all_stocks_code=np.array(self.data.code)
        choosen_stock_code=all_stocks_code[index]
        self.__clearOrCreate_table(conn,table_name)

        for code in choosen_stock_code:
            SQL='insert into %s select * from originData where code=\"%s\"' % (table_name,code)
            conn.execute(SQL)
        conn.commit()

        print('data in inserting completed')

if __name__=='__main__':
    autoenc=AutoEncoder()
    choosen_stock_index=autoenc.choose_stocks()

    autoenc.store_to_db(conn,choosen_stock_index,'autoencoder_choose_data')
    print('stock choosen completed! The choosen stocks\' index is:',' '.join(list(map(str,choosen_stock_index))))
    norm(conn,'autoencoder_choose_data','autoencoder_choose_norm')






