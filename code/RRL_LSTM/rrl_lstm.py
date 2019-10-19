import tensorflow as tf
import os
import configparser
from tensorflow.python.ops.rnn import dynamic_rnn

def getConfig(section, key):
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + '/config.ini'
    config.read(path)
    return config.get(section, key)

def tanh(input_, out_dim, scope=None, reuse=False):
    with tf.variable_scope(scope or 'tanh', reuse=reuse):
        W = tf.get_variable('W', [input_.get_shape()[1], out_dim])
        b = tf.get_variable('b', [out_dim])
    return tf.nn.tanh(tf.matmul(input_, W) + b)

class RRL(object):
    def __init__(self):
        # output dim=k choosen stock
        self.output_dim = int(getConfig('rrl', 'choosen_stocks_num'))
        self.window_len = int(getConfig('rrl', 'window_len'))
        self.stockNum = int(getConfig('data', 'stockNum'))
        # self.scope = scope

        self.batch_size = 1
        self.input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, self.output_dim * self.window_len])
        # self.rt = tf.placeholder(dtype=tf.float32, shape=[self.output_dim, None])  # self.T_num + self.window_len - 1
        self.price = tf.placeholder(dtype=tf.float32, shape=[self.output_dim, None])  # self.T_num+1
        self.index = tf.placeholder(dtype=tf.float32, shape=[None])  # shape=[self.T_num + 1]
        self.market_capitalization = tf.placeholder(dtype=tf.float32, shape=[self.output_dim, None])
        self.all_market_capitalization = tf.placeholder(dtype=tf.float32, shape=[self.stockNum, None])

        # Ftminus1: Initial portfolio weights
        self.Ftminus1 = tf.placeholder(dtype=tf.float32, shape=[self.output_dim])

        # train
        self.learning_rate = float(getConfig('rrl', 'learning_rate'))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def build(self):

        self.input = tf.nn.tanh(self.input)  #
        self.rnn = tf.nn.rnn_cell.LSTMCell(self.output_dim, activation=tf.nn.sigmoid, initializer=tf.initializers.random_normal())
        self.rnn_dr = tf.nn.rnn_cell.DropoutWrapper(self.rnn, input_keep_prob=0.9, output_keep_prob=1.0) # Dropout
        state = self.rnn.zero_state(batch_size=1,dtype=tf.float32)
        self.outputs, self.state = dynamic_rnn(self.rnn, self.input, initial_state=state, dtype=tf.float32)
        self.outputs = tf.nn.softmax(self.outputs) #
        self.origin_outputs = tf.squeeze(self.outputs)
        self.outputs = tf.concat([[self.Ftminus1],self.origin_outputs[0:-1,:]],axis=0)

        self.price_before = tf.transpose(self.price[:,0:-1]) #[step,stock_num]
        self.price_later = tf.transpose(self.price[:, 1:])  # [step,stock_num]
        # self.vol_before = tf.divide(tf.multiply(self.cash, self.outputs[0:-1,:]), self.price_before[0:-1,:])
        # self.vol_now = tf.divide(tf.multiply(self.cash, self.outputs[1:,:]), self.price_later[0:-1,:])
        # rt = tf.transpose(self.rt[:,self.window_len-1:]) #[step,stock_num]
        #Rt要改
        # self.returns = tf.reduce_sum(self.vol_before * rt,axis=1) - tf.reduce_sum(self.trans_ratio * self.price_later * tf.abs(self.vol_now - self.vol_before),axis=1)
        # self.returns = tf.log(tf.divide(tf.reduce_sum(tf.multiply(self.price_later[1:,:],self.outputs[1:,:]),axis=1),
        #                                 tf.reduce_sum(tf.multiply(self.price_before[1:,:],self.outputs[0:-1,:]),axis=1)))
        self.returns = tf.log(
            tf.divide(tf.reduce_sum(tf.multiply(self.price_later, self.outputs[1:, :]), axis=1),
                      tf.reduce_sum(tf.multiply(self.price_before, self.outputs[0:-1, :]), axis=1)))

        self.moments = tf.nn.moments(self.returns,axes=0)
        self.sharpe_ratio = self.moments[0] / tf.sqrt(self.moments[1])

        # 跟踪比率
        self.tracking_ratio = tf.divide(
            # tf.divide(tf.reduce_sum(self.all_market_capitalization[:,1:],axis=0),
            tf.divide(tf.reduce_sum(self.all_market_capitalization, axis=0),
                      tf.reduce_sum(self.all_market_capitalization[:,0],axis=0)),
            # tf.divide(tf.reduce_sum(tf.transpose(self.origin_outputs)*self.market_capitalization[:,1:],axis=0),
            tf.divide(tf.reduce_sum(tf.transpose(self.origin_outputs) * self.market_capitalization, axis=0),
                      tf.reduce_sum(tf.transpose(self.origin_outputs)*tf.expand_dims(self.market_capitalization[:,0],axis=1),axis=0))
        )

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.loss = -self.sharpe_ratio
        self.train_step_noholder = self.optimizer.minimize(self.loss)

        self.index_before =  tf.transpose(self.index[0:-1])
        self.index_after = tf.transpose(self.index[1:])

        # 超额收益
        # self.ER = tf.reduce_sum(tf.reduce_mean(tf.log(
        #     tf.reduce_sum(tf.multiply(tf.divide(self.price_later[1:,:], self.price_before[1:,:]), self.outputs[0:-1,:]),
        #                   axis=1)) - tf.log(tf.divide(self.index_after, self.index_before))))
        self.ER = tf.reduce_sum(tf.reduce_mean(
            tf.log(tf.divide(
                tf.reduce_sum(self.price_later* self.outputs[1:,:], axis=1),
                tf.reduce_sum(self.price_before * self.outputs[0:-1,:], axis=1))
            )
            - tf.log(tf.divide(self.index_after, self.index_before))
        ))
        # 跟踪误差
        self.TE = tf.reduce_mean(tf.sqrt(
            tf.reduce_sum(tf.square(
                tf.log(
                    # tf.reduce_sum(tf.multiply(tf.divide(self.price_later, self.price_before), self.vol_now),axis=1)) -
                    tf.divide(tf.reduce_sum(self.price_later * self.outputs[1:,:], axis=1),
                              tf.reduce_sum(self.price_before * self.outputs[0:-1,:], axis=1))) -
                tf.log(tf.divide(self.index_after, self.index_before))))))
        # Information Ratio(IR)
        self.IR = tf.divide(self.ER,self.TE)