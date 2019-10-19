import tensorflow as tf
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from read_config import *
import Portfolio_ddpg.env_ours

#####################  hyper parameters  ####################

MAX_EPISODES = 1000
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002   # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.1      # soft replacement
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.win_len = int(config.get('ddpg', 'win_len'))
        self.stock_num = int(config.get('data', 'stock_choose_num'))
        self.feature_num = int(config.get('data', 'feature_num'))

        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [BATCH_SIZE, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [BATCH_SIZE, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        #self.at_params = tf.get_collection(scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        #self.ct_params = tf.get_collection(scope='Critic/target')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement（更新慢但稳定）
        #用eval的参数更新target的参数
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())

        self.soft_replace_ = [tf.assign(t, e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
        self.sess.run(self.soft_replace_)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: np.tile(s[np.newaxis, :].reshape(1,self.s_dim),(BATCH_SIZE,1))})[0]

    def learn(self):
        #saver = tf.train.Saver()
        # soft target replacement
        self.sess.run(self.soft_replace)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        #print("bs:",bs.shape)
        #print('ba:',ba.shape)
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        #saver.save(self.sess,'/model.ckpt')

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a.reshape([1,self.stock_num]), np.array([r]).reshape([1,1]), s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            #print("as:",s.shape)
            net = tf.layers.dense(s, 15, activation=tf.nn.relu, name='l1', trainable=trainable)
            #a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            #return tf.multiply(a, self.a_bound, name='scaled_a')
            #将激活函数改成sigmoid，则不需要乘a_bound调整范围。
            print(a.shape)
            return a
    """
    def _build_a(self, s, scope, trainable):
        s_ = s[:,:self.s_dim-self.stock_num]
        s_ = tf.reshape(s_,[-1,self.win_len,self.feature_num*self.stock_num])
        with tf.variable_scope(scope):
            lstm_cell = rnn.BasicLSTMCell(num_units=self.stock_num)
            init_state = lstm_cell.zero_state(BATCH_SIZE,dtype=tf.float32)
            outputs,_= tf.nn.dynamic_rnn(lstm_cell,inputs = s_,initial_state = init_state,time_major = False)
            output_rnn = tf.nn.softmax(outputs)
            return output_rnn[:,-1,:]
    """

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 15
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################
env = Portfolio_ddpg.env_ours.Env()

s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 100  # control exploration
t1 = time.time()

x=[]
x_=[]

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        # Add exploration noise
        a = ddpg.choose_action(s)
        #print("a:",a)
        a = np.clip(np.random.normal(a, var), 0, 1)    # add randomness to action selection for exploration
        #a = np.clip(a,0,1)
        a_ = a.copy()
        #限制数组a和为1
        for index in range(len(a)):
            a_[index] = a[index]/(sum(a)+1e-5)
        #print("a_:",a_)
        s_, r, done = env.step(s,a_)
        if(done == False):
            #print("False")
            s = env.reset()
        else:
            #ddpg.store_transition(s, a, r / 10, s_)
            ddpg.store_transition(s, a, r, s_)
            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()
        s = s_
        ep_reward += r

        x.append(i)
        x_.append(ep_reward)

        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward:',ep_reward, 'Explore: %.2f' % var, )
            #print('Episode:', i, ' Reward:', ep_reward )
            # if ep_reward > -300:RENDER = True
            break

plt.plot(x,x_)
plt.show()

#save model
saver = tf.train.Saver()
saver.save(ddpg.sess,'./model.ckpt')

list=[]
list.append(env.codelist)

list0_ = [0 for i in range(env.stock_num)]
for i in range(env.win_len-1):
    list.append(list0_)
list1_ = [0.1 for i in range(env.stock_num)]
list.append(list1_)

with tf.Session() as sess:
    saver.restore(sess,'./model.ckpt')
    init_weights = list[-1]
    s = env.test_data[:,:env.win_len,:env.feature_num].reshape(1, env.state_dim-env.stock_num)
    #s = env.test_data[:, :env.win_len, env.feature_num:].reshape(1, env.state_dim - env.stock_num)
    s = np.concatenate([s,np.array(init_weights).reshape(1,env.stock_num)],axis=1)
    for i in range(env.win_len,60):
        #a = sess.run(tf.get_default_graph().get_tensor_by_name("Actor/eval/a/Sigmoid:0"),feed_dict={ddpg.S:s})
        a = sess.run(tf.get_default_graph().get_tensor_by_name("Actor/eval/a/Sigmoid:0"), feed_dict={ddpg.S: np.tile(s,(BATCH_SIZE,1))})
        # a = a[:,-1,:]
        s_ = env.test_data[:, i - env.win_len+1:i+1, :env.feature_num].reshape(1, env.state_dim-env.stock_num)
        #s_ = env.test_data[:, i - env.win_len + 1:i + 1, env.feature_num:].reshape(1, env.state_dim - env.stock_num)
        #print(a.shape)
        a = a[0,:].reshape(1,env.stock_num)
        s_ = np.concatenate([s_,a],axis=1)
        #print(a)
        a_ = a[0].copy()
        # 限制数组a和为1
        for index in range(len(a[0])):
            a_[index] = a[0][index] / (sum(a[0]) + 1e-5)
        list.append(a_)
        s = s_

df = pd.DataFrame(list)
df.to_excel('F:\portfolio_rx\portfolio_rx\DJ30\\vol_ddpg_result.xls')


#print('Running time: ', time.time() - t1)