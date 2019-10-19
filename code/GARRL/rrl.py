import tensorflow as tf
import configparser
import os
def getConfig(section, key):
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + '/config.ini'
    config.read(path,encoding='utf-8')
    return config.get(section, key)

class RRL(object):
    def __init__(self,scope):
        # The output dimension is the number of stocks selected
        self.output_dim=int(getConfig('rrl','choosen_stocks_num'))
        # The window length is the length of the time range that needs to be considered at each moment
        self.window_len=int(getConfig('rrl','window_len'))
        # Number of indicators to consider
        self.indicator_num = int(getConfig('rrl', 'indicator_num'))
        self.stockNum = int(getConfig('data', 'stockNum'))
        self.scope=scope
        # Parameter initialization
        self.stddev=float(getConfig('rrl', 'stddev'))
        self.initializer=tf.initializers.random_normal(stddev=self.stddev)
        self.batch_size=1
        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=[self.batch_size, None,self.output_dim * self.window_len * self.indicator_num])
        self.price = tf.placeholder(dtype=tf.float32, shape=[self.output_dim, None])#None:1+ total length of time
        self.index=tf.placeholder(dtype=tf.float32,shape=[None])#shape=[1+ total length of time]
        #self.Ftminus1 means Initial portfolio weights
        self.Ftminus1=tf.placeholder(dtype=tf.float32,shape=[self.output_dim])
        self.cell_state = tf.placeholder(tf.float32, [1, self.output_dim])
        self.hidden_state = tf.placeholder(tf.float32, [1, self.output_dim])
        self.init_state = tf.nn.rnn_cell.LSTMStateTuple(self.cell_state, self.hidden_state)
        #train
        self.learning_rate = float(getConfig('rrl', 'learning_rate'))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.build()

    def build(self):
        with tf.variable_scope(self.scope, reuse=None, initializer=self.initializer):
            cell = tf.nn.rnn_cell.LSTMCell(self.output_dim)
            outputs, self.state = tf.nn.dynamic_rnn(cell, self.input,initial_state=self.init_state,dtype=tf.float32)
            outputs=tf.nn.softmax(outputs)
            self.origin_outputs=tf.squeeze(outputs)
            # The last decision is useless, so get rid of it.
            # [step,stock]
            self.outputs=tf.concat([[self.Ftminus1],self.origin_outputs[0:-1,:]],axis=0)
            price_before=tf.transpose(self.price[:,0:-1]) #[step,stock_num] 从第0天开始
            price_later = tf.transpose(self.price[:, 1:])  # [step,stock_num] 从第1天开始
            self.rt = tf.log(tf.divide(tf.reduce_sum(
                price_later*self.outputs[1:,:],axis=1),tf.reduce_sum(price_before*self.outputs[0:-1,:],axis=1)))
            self.moments = tf.nn.moments(self.rt,axes=0)
            self.sharpe_ratio = self.moments[0] / tf.sqrt(self.moments[1])
            index_before = tf.transpose(self.index[0:-1])
            index_after = tf.transpose(self.index[1:])
            self.RT = tf.log(tf.divide(index_after, index_before))
            # Excess return
            ER = tf.reduce_mean(self.rt - self.RT)
            self.ER = tf.reduce_mean(
                tf.log(tf.reduce_sum(tf.divide(
                    price_later, price_before) * self.outputs[1:, :], axis=1)) - self.RT)
            # tracking error
            self.TE = 1.0 / self.output_dim * (tf.sqrt(tf.reduce_sum(tf.square(self.rt - self.RT))))
            # self.loss=0.2*self.TE-0.8*ER
            self.loss = -self.sharpe_ratio
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            grads_all = self.optimizer.compute_gradients(self.loss)
            grads_vars = [v for (g, v) in grads_all if g is not None]
            self.grads = self.optimizer.compute_gradients(self.loss, grads_vars)
            self.grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v) for (g, v) in self.grads]
            self.train_step_holder = self.optimizer.apply_gradients(self.grads_holder)
            self.train_step_noholder = self.optimizer.apply_gradients(self.grads)
            self.vars = tf.get_collection(tf.GraphKeys.VARIABLES,scope=self.scope)
            self.var_holder = [tf.placeholder(tf.float32, shape=g.shape) for g in self.vars]
            self.var_num = len(self.var_holder)
            self.assign_step = [tf.assign(v, v_h) for (v, v_h) in zip(self.vars, self.var_holder)]