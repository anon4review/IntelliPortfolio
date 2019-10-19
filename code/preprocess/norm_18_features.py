import numpy as np
FEATURE_NUM=18

def clearOrCreate_table(conn,name):
    SQL = 'create table if not exists %s (code string,date date,openprice double,' \
          'maxprice double,minprice double, closeprice double,vol double,money double,updown double,' \
          'updownratio double,meanprice double, TurnoverRate double,CirculationValue double,TotalValue double,' \
          'FlowEquity double,TotalEquity double,PE double,PB double,P2S double,PCF double,closepriceorigin double)' % name
    conn.execute(SQL)
    SQL='delete from %s'%name
    conn.execute(SQL)
    print('table '+str(name)+' create ready!')

def norm(conn,origin_table_name,norm_table_name):
    #normlization
    clearOrCreate_table(conn,norm_table_name)
    maxFeatures = np.zeros([FEATURE_NUM])
    minFeatures = np.zeros([FEATURE_NUM])
    # Maximum of 18 features
    SQL = 'select max(openprice),max(maxprice),max(minprice),max(closeprice),max(vol),max(money),max(updown),max(updownratio)' \
          ',max(meanprice),max(TurnoverRate),max(CirculationValue),max(TotalValue),max(FlowEquity),max(TotalEquity)' \
          ',max(PE),max(PB),max(P2S),max(PCF) from %s'%origin_table_name
    cursor = conn.execute(SQL)
    result = cursor.fetchall()[0]
    for i in range(FEATURE_NUM):
        maxFeatures[i] = result[i]

    # Minimum of 18 features
    SQL = 'select min(openprice),min(maxprice),min(minprice),min(closeprice),min(vol),min(money),min(updown),min(updownratio)' \
          ',min(meanprice),min(TurnoverRate),min(CirculationValue),min(TotalValue),min(FlowEquity),min(TotalEquity)' \
          ',min(PE),min(PB),min(P2S),min(PCF) from %s'%origin_table_name
    cursor = conn.execute(SQL)
    result = cursor.fetchall()[0]
    for i in range(FEATURE_NUM):
        minFeatures[i] = result[i]

    # Circulating to generate normalized data and inserting it into the database
    SQL = 'select * from %s'%origin_table_name
    cursor = conn.execute(SQL)
    result = cursor.fetchall()
    i = 0
    for row in result:
        print(i)
        i += 1
        normalization_data = (list(row)[3:] - minFeatures) / (maxFeatures - minFeatures)  # 3
        closeprice = list(row)[6]
        SQL = 'insert into %s values(\"%s\",%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf' \
              ',%lf,%lf,%lf,%lf,%lf)' % (norm_table_name,row[0], int(row[1]), normalization_data[0], normalization_data[1]
                                     , normalization_data[2], normalization_data[3], normalization_data[4]
                                     , normalization_data[5], normalization_data[6], normalization_data[7]
                                     , normalization_data[8], normalization_data[9], normalization_data[10]
                                     , normalization_data[11], normalization_data[12], normalization_data[13]
                                     , normalization_data[14], normalization_data[15], normalization_data[16]
                                     , normalization_data[17],closeprice)
        conn.execute(SQL)
        conn.commit()
    conn.close()
    print('random choose and normlize completed')