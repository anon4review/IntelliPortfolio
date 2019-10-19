import tensorflow as tf
import numpy as np
from portfolio_LSTM.read_config import *
from portfolio_LSTM.dataKeeper import DataKeeper
import pandas as pd
class Portfolio(object):
    def __init__(self):
        init_decision=np.array([1.0/CHOOSEN_STOCK_NUM for _ in range(CHOOSEN_STOCK_NUM)])
        self.init_decisons=np.tile(init_decision,[BATCH_SIZE,1,1])
        self.datakeeper=DataKeeper()
        self.sess=tf.Session()
        self.build()
        self.sess.run(tf.initialize_all_variables())

    def build(self):
        with tf.variable_scope('actor',initializer=tf.initializers.random_normal(0.0,0.1)):
            self.X = tf.placeholder(tf.float32, [BATCH_SIZE, None, CHOOSEN_STOCK_NUM*WIN_LEN*FEATURE_NUM+1*WIN_LEN])
            self.value=tf.placeholder(tf.float32,[BATCH_SIZE,None,CHOOSEN_STOCK_NUM])
            self.index=tf.placeholder(tf.float32,[BATCH_SIZE,None])

            cell = tf.nn.rnn_cell.BasicLSTMCell(CHOOSEN_STOCK_NUM)
            init_state=cell.zero_state(BATCH_SIZE,dtype=tf.float32)
            output_rnn, _ = tf.nn.dynamic_rnn(cell, self.X,initial_state=init_state ,dtype=tf.float32)#output:[batch,steps,decision]
            self.output_rnn=tf.nn.softmax(output_rnn)
            self.batch_decisions_with0=tf.concat([self.init_decisons,self.output_rnn],axis=1)
            batch_decisions_before=self.batch_decisions_with0[:,:-1,:]
            batch_decisions_after=self.batch_decisions_with0[:,1:,:]
            batch_price_before=self.value[:,0:-1,:]
            batch_price_after=self.value[:,1:,:]
            batch_index_before=self.index[:,0:-1]
            batch_index_after=self.index[:,1:]

            #compute TE ER
            #[batch,step]
            self.rt = tf.log(tf.divide(tf.reduce_sum(batch_price_after * batch_decisions_after, axis=2),
                                  tf.reduce_sum(batch_price_before * batch_decisions_before, axis=2)))
            #[batch,step]
            Rt = tf.log(tf.divide(batch_index_after, batch_index_before))

            # Tracking Error
            #[batch]
            self.TE = (1.0/DECISION_DAY_NUM)*tf.sqrt(tf.reduce_sum(tf.square(self.rt - Rt),axis=1))

            # Excess Return
            # 提取权重
            #[batch]
            self.ER = tf.reduce_mean(self.rt- Rt,axis=1)

            # Sharpe Ratio
            self.mean,self.var= tf.nn.moments(self.rt,axes=1)
            self.sharpe_ratio = tf.reduce_mean(self.mean / tf.sqrt(self.var))
            self.cumulative_rt=tf.reduce_mean(tf.reduce_sum(self.rt,axis=1))

            self.loss=tf.reduce_mean(LAMDA * self.TE - (1.0-LAMDA) * self.ER)
            self.train_step = tf.train.AdamOptimizer(LR).minimize(self.loss)

    def train_one_time(self):
        batch_train_data,batch_value_data,batch_index_data=self.datakeeper.get_random_batch()
        _,weights,loss,TE,ER,SR=self.sess.run([self.train_step,self.batch_decisions_with0,self.loss,self.TE,self.ER,self.sharpe_ratio],feed_dict={self.X:batch_train_data,
                                                 self.value:batch_value_data,
                                                 self.index:batch_index_data })
        return loss,weights,TE,ER,SR

    def test(self):
        test_data,test_value,test_index=self.datakeeper.get_test_data()
        weights,loss,TE,ER,SR=self.sess.run([self.output_rnn,self.loss,self.TE,self.ER,self.sharpe_ratio],feed_dict={self.X:test_data,
                                                                                                      self.value: test_value,
                                                                                                      self.index:test_index
                                                                                          })
        return weights,loss,TE[0],ER[0],SR

    def val(self):
        val_data,val_alue,val_index=self.datakeeper.get_val_data()
        weights,loss,TE,ER,SR=self.sess.run([self.output_rnn,self.loss,self.TE,self.ER,self.sharpe_ratio],feed_dict={self.X:val_data,
                                                                                                      self.value: val_alue,
                                                                                                      self.index:val_index
                                                                                          })
        return weights,loss,TE[0],ER[0],SR

if __name__ == '__main__':
    rl = Portfolio()
    for epoch in range(MAX_EPOCH):
        loss,weights,TE,ER,SR=rl.train_one_time()
        if epoch%100==0:
            print('EPOCH '+str(epoch+1)+': ')
            print('===================train=====================')
            print('loss:',loss)
            print('SR:'+str(SR))
            print('TE:' + str(TE[0]))
            print('ER:' + str(ER[0]))


    print('===================VAL=====================')
    val_weights, loss, TE, ER, SR = rl.val()
    print('TE:', TE)
    print('ER:', ER)
    print('SR:', SR)
    # 2 excel
    list_2excel = []
    list_2excel.append(rl.datakeeper.codelist)
    for i in range(WIN_LEN - 1):
        temp = []
        for i in range(rl.datakeeper.codelist.__len__()):
            temp.append(0.0)
        list_2excel.append(temp)
    temp = []
    for i in range(rl.datakeeper.codelist.__len__()):
        temp.append(1.0 / rl.datakeeper.codelist.__len__())
    list_2excel.append(temp)
    for action in val_weights[0]:
        list_2excel.append(action)
    df = pd.DataFrame(list_2excel)
    df.to_excel('..\9test\\val_set_later120.xlsx')

    #test
    print('===================test=====================')
    test_weights,loss,TE,ER,SR=rl.test()
    print('SR:',SR)
    print('TE:',TE)
    print('ER:',ER)
    # 2 excel
    list_2excel = []
    list_2excel.append(rl.datakeeper.codelist)
    for i in range(WIN_LEN - 1):
        temp = []
        for i in range(rl.datakeeper.codelist.__len__()):
            temp.append(0.0)
        list_2excel.append(temp)
    temp = []
    for i in range(rl.datakeeper.codelist.__len__()):
        temp.append(1.0 / rl.datakeeper.codelist.__len__())
    list_2excel.append(temp)
    for action in test_weights[0]:
        list_2excel.append(action)
    df = pd.DataFrame(list_2excel)
    df.to_excel('..\9test\\test_set_later120.xlsx')


