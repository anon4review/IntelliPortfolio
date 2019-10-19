import sqlite3
import configparser
import os

def getConfig(section, key):
    config = configparser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + '/config.ini'
    config.read(path)
    return config.get(section, key)

db_path=getConfig('db', 'db_path')
conn = sqlite3.connect(database=db_path)

def clearOrCreate_table(name):
    SQL = 'create table if not exists %s (date date,closeprice double)' % name
    conn.execute(SQL)
    SQL='delete from %s'%name
    conn.execute(SQL)
    print('table '+str(name)+' create ready!')

#normlization
clearOrCreate_table('indexNorm')

SQL = 'select max(closeprice) from indexes'
cursor = conn.execute(SQL)
result = cursor.fetchall()[0]
maxindex = result[0]

SQL = 'select min(closeprice) from indexes'
cursor = conn.execute(SQL)
result = cursor.fetchall()[0]
minindex = result[0]

# Circulating to generate normalized data and inserting it into the database
SQL = 'select * from indexes'
cursor = conn.execute(SQL)
result = cursor.fetchall()
i = 0
for row in result:
    print(i)
    i += 1
    normalization_data = (row[1] - minindex) / (maxindex - minindex)
    SQL = 'insert into indexNorm values(%d,%lf)' % (row[0],normalization_data)
    conn.execute(SQL)
    conn.commit()
conn.close()