import sqlite3
import random
from preprocess.norm_18_features import norm
CHOOSE_NUM = 10
FEATURE_NUM=18
import configparser
import os

def getConfig(section, key):
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + '/config.ini'
    config.read(path)
    return config.get(section, key)


def clearOrCreate_table(conn,name):
    SQL = 'create table if not exists %s (code string,date date,openprice double,' \
              'maxprice double,minprice double, closeprice double,vol double,money double,updown double,' \
              'updownratio double,meanprice double, TurnoverRate double,CirculationValue double,TotalValue double,' \
              'FlowEquity double,TotalEquity double,PE double,PB double,P2S double,PCF double)'%name
    # SQL = 'create table if not exists %s (code string,date date,openprice double,' \
    #       'maxprice double,minprice double, closeprice double,vol double,adjcloseprice double)' % name
    print(SQL)
    conn.execute(SQL)
    SQL='delete from %s'%name
    conn.execute(SQL)
    print('table '+str(name)+' create ready!')

if __name__ == '__main__':
    db_path = getConfig('db', 'db_path')
    conn = sqlite3.connect(database=db_path)
    #select stocks
    clearOrCreate_table(conn,'random_choose_data')
    code=[]
    SQL = 'select distinct code from originData'
    cursor = conn.execute(SQL)
    for row in cursor:
        code.append(row[0])
    choose_code = random.sample(code,CHOOSE_NUM)
    print(choose_code)

    for element in choose_code:
        SQL = 'insert into random_choose_data select * from originData where code = \"%s\"' %element
        conn.execute(SQL)
    print('random choose stocks finished!')

    #normlization
    norm(conn,'random_choose_data','random_choose_norm')
    conn.close()