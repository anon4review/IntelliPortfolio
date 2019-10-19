import numpy as np
import configparser
import sqlite3

config=configparser.ConfigParser()
config.read('./config.ini')
feature_num=int(config.get('data','feature_num'))
db_path=config.get('db','db_path')
maxFeatures=np.zeros([feature_num])
minFeatures=np.zeros([feature_num])

conn = sqlite3.connect(db_path)
# Maximum of 18 features
SQL='select max(openprice),max(maxprice),max(minprice),max(closeprice),max(vol),max(money),max(updown),max(updownratio)' \
    ',max(meanprice),max(TurnoverRate),max(CirculationValue),max(TotalValue),max(FlowEquity),max(TotalEquity)' \
    ',max(PE),max(PB),max(P2S),max(PCF) from originData'
#SQL='select max(openprice),max(maxprice),max(minprice),max(closeprice),max(vol),max(adjcloseprice) from originDataByVol'
cursor=conn.execute(SQL)
result=cursor.fetchall()[0]
for i in range(feature_num):
    maxFeatures[i]=result[i]

# Minimum of 18 features
#SQL='select min(openprice),min(maxprice),min(minprice),min(closeprice),min(vol),min(adjcloseprice) from DJoriginalData'
SQL='select min(openprice),min(maxprice),min(minprice),min(closeprice),min(vol),min(money),min(updown),min(updownratio)' \
    ',min(meanprice),min(TurnoverRate),min(CirculationValue),min(TotalValue),min(FlowEquity),min(TotalEquity)' \
    ',min(PE),min(PB),min(P2S),min(PCF) from originData'
cursor=conn.execute(SQL)
result=cursor.fetchall()[0]
for i in range(feature_num):
    minFeatures[i]=result[i]

#Circulating to generate normalized data and inserting it into the database
SQL='select * from originData'
cursor=conn.execute(SQL)
result=cursor.fetchall()
i=0
for row in result:
    print(i)
    i+=1
    normalization_data=(list(row)[2:]-minFeatures)/(maxFeatures-minFeatures)#3
    SQL='insert into nor_data values(\"%s\",%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf' \
        ',%lf,%lf,%lf,%lf)' %(row[0],int(row[1]),normalization_data[0],normalization_data[1]
                                      ,normalization_data[2],normalization_data[3],normalization_data[4]
                                      ,normalization_data[5],normalization_data[6],normalization_data[7]
                                      ,normalization_data[8],normalization_data[9],normalization_data[10]
                                      ,normalization_data[11],normalization_data[12],normalization_data[13]
                                      ,normalization_data[14],normalization_data[15],normalization_data[16]
                                      ,normalization_data[17])
    # SQL = 'insert into DJnorm values(\"%s\",%d,%lf,%lf,%lf,%lf,%lf,%lf)' % (row[0], row[1], normalization_data[0], normalization_data[1]
    #                              , normalization_data[2], normalization_data[3], normalization_data[4]
    #                              , normalization_data[5])
    conn.execute(SQL)
conn.commit()