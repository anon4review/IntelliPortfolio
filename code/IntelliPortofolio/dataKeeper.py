import numpy as np
from portfolio_LSTM.read_config import *
import sqlite3
import random
class DataKeeper(object):
    def __init__(self):
        conn = sqlite3.connect(DB_PATH)

        #get choosen stocks' code
        self.codelist = []
        SQL = "select distinct code from %s" %  TABLE_NAME
        cursor = conn.execute(SQL)
        for row in cursor:
            self.codelist.append(row[0])
        print(self.codelist)

        #get all data from db
        result = np.array(
            [[[0.0 for _ in range(FEATURE_NUM + 1)] for _ in range(ALL_DAY_NUM)] for _ in range(CHOOSEN_STOCK_NUM)])
        for stock_i,code in enumerate(self.codelist):
            SQL = 'select * from %s where code=\'%s\'' % (TABLE_NAME,code)
            cursor = conn.execute(SQL)
            for date_i,row in enumerate(cursor):
                result[stock_i][date_i] = row[2:]

        #get all index_norm from db
        self.index = np.array([0.0 for _ in range(ALL_DAY_NUM)])
        SQL = 'select closeprice from indexes'
        cursor = conn.execute(SQL)
        for date_i,row in enumerate(cursor):
            self.index[date_i] = row[0]

        # get all index from db
        self.index_feature = np.array([0.0 for _ in range(ALL_DAY_NUM)])
        SQL = 'select closeprice from indexNorm'
        cursor = conn.execute(SQL)
        for date_i, row in enumerate(cursor):
            self.index_feature[date_i] = row[0]

        #get train data
        self.train_data=result[:,:TRAIN_NUM,:]

        # get val data
        self.val_data = result[:, TRAIN_NUM:TRAIN_NUM+VAL_NUM, :]

        #get test data
        self.test_data=result[:,TRAIN_NUM+VAL_NUM:,:]
        # self.test_data = result[:, TRAIN_NUM:, :]

    def get_random_batch(self):
        batch_train_data=[]
        batch_value_data=[]
        batch_index_data = []
        for i in range(BATCH_SIZE):
            #index+win_len+decision_num-1<=train_num
            random_begin_index=random.randint(0,TRAIN_NUM-DECISION_DAY_NUM-WIN_LEN)
            #print(random_begin_index)
            one_batch_train_data = []
            one_batch_value_data=[]
            one_batch_index_data = []
            j=0
            while(True):
                one_step_index_data = self.index[random_begin_index + WIN_LEN + j - 1]
                one_batch_index_data.append(one_step_index_data)

                one_step_train_data=list(self.train_data[:,random_begin_index+j:random_begin_index+WIN_LEN+j,:FEATURE_NUM]
                                   .reshape(CHOOSEN_STOCK_NUM*WIN_LEN*FEATURE_NUM))
                for k in range(WIN_LEN):
                    one_step_train_data.append(self.index_feature[random_begin_index + k + j ]) #cosider index
                one_batch_train_data.append(one_step_train_data)

                one_step_value_data=list(self.train_data[:,random_begin_index+WIN_LEN+j-1,-1]
                                         .reshape(CHOOSEN_STOCK_NUM))
                one_batch_value_data.append(one_step_value_data)

                j+=1
                if j==DECISION_DAY_NUM:
                    break
            batch_train_data.append(one_batch_train_data)

            one_step_value_data = list(self.train_data[:, random_begin_index + WIN_LEN + j - 1, -1]
                                       .reshape(CHOOSEN_STOCK_NUM))
            one_batch_value_data.append(one_step_value_data)
            one_step_index_data = self.index[random_begin_index + WIN_LEN + j - 1]
            one_batch_index_data.append(one_step_index_data)
            batch_value_data.append(one_batch_value_data)
            batch_index_data.append(one_batch_index_data)
        return np.array(batch_train_data),np.array(batch_value_data),np.array(batch_index_data)

    def get_test_data(self):
        test_data=[]
        test_value=[]
        test_index=[]
        i=0
        while True:
            temp_index = self.index[TRAIN_NUM  + i + WIN_LEN+ VAL_NUM - 1]#+ VAL_NUM
            test_index.append(temp_index)

            temp_data=list(self.test_data[:,i:i+WIN_LEN,:FEATURE_NUM].reshape(CHOOSEN_STOCK_NUM*WIN_LEN*FEATURE_NUM))
            for k in range(WIN_LEN):
                temp_data.append(self.index_feature[TRAIN_NUM + VAL_NUM + i +k])#+ VAL_NUM
            test_data.append(temp_data)

            temp_value = list(self.test_data[:, i + WIN_LEN-1, -1].reshape(CHOOSEN_STOCK_NUM))
            test_value.append(temp_value)

            i += 1
            if i == TEST_NUM-WIN_LEN:
                break
        temp_value = list(self.test_data[:, i + WIN_LEN-1, -1].reshape(CHOOSEN_STOCK_NUM))
        test_value.append(temp_value)
        temp_index = self.index[TRAIN_NUM + i + WIN_LEN +VAL_NUM- 1]#+VAL_NUM
        test_index.append(temp_index)

        test_data=np.array(test_data).reshape([1,TEST_NUM-WIN_LEN,CHOOSEN_STOCK_NUM*WIN_LEN*FEATURE_NUM+1*WIN_LEN])
        test_data=np.tile(test_data,[BATCH_SIZE,1,1])
        test_value = np.array(test_value).reshape([1, TEST_NUM-WIN_LEN+1 , CHOOSEN_STOCK_NUM])
        test_value = np.tile(test_value, [BATCH_SIZE, 1, 1])
        test_index = np.array(test_index).reshape([1, TEST_NUM-WIN_LEN+1 ])
        test_index = np.tile(test_index, [BATCH_SIZE, 1])

        return test_data,test_value,test_index

    def get_val_data(self):
        val_data=[]
        val_value=[]
        val_index=[]
        i=0
        while True:
            temp_index = self.index[TRAIN_NUM + i + WIN_LEN - 1]
            val_index.append(temp_index)

            temp_data=list(self.val_data[:,i:i+WIN_LEN,:FEATURE_NUM].reshape(CHOOSEN_STOCK_NUM*WIN_LEN*FEATURE_NUM))
            for k in range(WIN_LEN):
                temp_data.append(self.index_feature[TRAIN_NUM + i + k])
            val_data.append(temp_data)

            temp_value = list(self.val_data[:, i + WIN_LEN-1, -1].reshape(CHOOSEN_STOCK_NUM))
            val_value.append(temp_value)

            i += 1
            if i == VAL_NUM-WIN_LEN:
                break
        temp_value = list(self.val_data[:, i + WIN_LEN-1, -1].reshape(CHOOSEN_STOCK_NUM))
        val_value.append(temp_value)
        temp_index = self.index[TRAIN_NUM + i + WIN_LEN - 1]
        val_index.append(temp_index)

        val_data=np.array(val_data).reshape([1,VAL_NUM-WIN_LEN,CHOOSEN_STOCK_NUM*WIN_LEN*FEATURE_NUM+1*WIN_LEN])
        val_data=np.tile(val_data,[BATCH_SIZE,1,1])
        val_value = np.array(val_value).reshape([1, VAL_NUM-WIN_LEN+1 , CHOOSEN_STOCK_NUM])
        val_value = np.tile(val_value, [BATCH_SIZE, 1, 1])
        val_index = np.array(val_index).reshape([1, VAL_NUM-WIN_LEN+1 ])
        val_index = np.tile(val_index, [BATCH_SIZE, 1])

        return val_data,val_value,val_index



if __name__ == '__main__':
    datakeep=DataKeeper()
    batch_train_data,batch_value_data,batch_index_data=datakeep.get_random_batch()
    test_data,test_value,test_index=datakeep.get_test_data()
    # print(test_data.shape)
    print(test_value)
    # print(test_index.shape)

    # print(batch_train_data)
    # print(batch_value_data)
    # print(batch_index_data)





