import sqlite3
from preprocess.norm_18_features import norm

BEGIN_DATE_VOL_SEL=1515081600
END_DATE_VOL_SEL=1530720000
CHOOSEN_STOCK_NUM=10

import configparser
import os

def getConfig(section, key):
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + '/config.ini'
    config.read(path)
    return config.get(section, key)
db_path = getConfig('db', 'db_path')
conn = sqlite3.connect(database=db_path)

code=[]

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

#get stock code and name
SQL = "select code,sum(vol) from originData where date >=%d and date<=%d group by code order by sum(vol) DESC limit %d" % (BEGIN_DATE_VOL_SEL,END_DATE_VOL_SEL,CHOOSEN_STOCK_NUM)
cursor = conn.execute(SQL)
data=cursor.fetchall()
for row in data:
    code.append(row[0])
print(code)

clearOrCreate_table(conn,'vol_choosen_data')
for co in code:
    print(co)
    SQL = 'insert into vol_choosen_data select * from originData where code=\"%s\"' % co
    conn.execute(SQL)

norm(conn,'vol_choosen_data','vol_choosen_norm')


