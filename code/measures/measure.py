import xlrd
import numpy as np
from measures.read_config import *
import sqlite3

data = xlrd.open_workbook("F:\portfolio_rx\portfolio_rx\\N225\\heuristicGA_result.xlsx")
table = data.sheets()[0]
nrows = table.nrows
ncols = table.ncols

list = []
stock = []
all_stock = []

for row in range(1,nrows):
    stock.append(table.cell(row, 0).value)
print(stock)

for row in range(1,nrows):
    rowtemp = []
    for col in range(1,ncols):
        # rowtemp.append(float(table.cell(row, col).value))
        rowtemp.append(table.cell(row, col).value)
    list.append(rowtemp)
weight = np.array(list)
if ncols == 2:
    weight = np.tile(weight, (1, 60 - WIN_LEN +1))
else:
    # weight = weight[:, window_len:]
    weight = weight[:, WIN_LEN - 1:]
print(weight)

price = np.array([[0.0 for _ in range(60)] for _ in range(CHOOSEN_STOCK_NUM)])
index = np.array([0.0 for _ in range(60)])
# all_market_capitalization = np.array([[0.0 for _ in range(60)] for _ in range(STOCK_NUM)])
# market_capitalization = np.array([[0.0 for _ in range(60)] for _ in range(CHOOSEN_STOCK_NUM)])


conn = sqlite3.connect(DB_PATH)

# get price
stock_i = 0
for code in stock:
    SQL = "select closeprice from originData where code = \'%s\' and date >= %d and date <= %d" % (code,BEGIN_DATE,END_DATE)
    cursor = conn.execute(SQL)
    data = cursor.fetchall()
    date_i = 0
    for row in data:
        price[stock_i][date_i] = row[0]
        date_i += 1
    stock_i += 1
# price = price[:, window_len:]
price = price[:, WIN_LEN - 1:]

# get index
SQL = "select closeprice from indexes where date >= %d and date <= %d"% (BEGIN_DATE,END_DATE)
cursor = conn.execute(SQL)
data = cursor.fetchall()
date_i = 0
for row in data:
    index[date_i] = row[0]
    date_i += 1
# index = index[window_len:]
index = index[WIN_LEN - 1:]

# # get market_capitalization(choosen stock TotalValue)
# stock_i = 0
# for code in stock:
#     SQL = 'select TotalValue from originData where code=\'%s\' and date >= 1531238400 and date <= 1538496000' % code
#     cursor = conn.execute(SQL)
#     data = cursor.fetchall()
#     date_i = 0
#     for row in data:
#         market_capitalization[stock_i][date_i] = row[0]
#         date_i += 1
#     stock_i += 1
# # market_capitalization = market_capitalization[:, window_len:]
# market_capitalization = market_capitalization[:, WIN_LEN - 1:]

# # get all stock
# SQL = "select distinct code from originData"
# cursor = conn.execute(SQL)
# data = cursor.fetchall()
# for row in data:
#     all_stock.append(row[0])

# # get all_market_capitalization(all stock TotalValue)
# stock_i = 0
# for code in all_stock:
#     SQL = 'select TotalValue from originData where code=\'%s\' and date >= 1531238400 and date <= 1538496000' % code
#     cursor = conn.execute(SQL)
#     data = cursor.fetchall()
#     date_i = 0
#     for row in data:
#         all_market_capitalization[stock_i][date_i] = row[0]
#         date_i += 1
#     stock_i += 1
# # all_market_capitalization = all_market_capitalization[:, window_len:]
# all_market_capitalization = all_market_capitalization[:, WIN_LEN - 1:]

price_before = price[:, 0:-1]
price_later = price[:, 1:]


index_before =  index[0:-1]
index_after = index[1:]



Rt = np.log(np.divide(index_after, index_before))

# rt
# 提取权重
# rt = np.log(np.sum(weight[:,1:] * np.divide(price_later,price_before), axis=0))
# 两个权重
rt = np.log(np.divide(np.sum(price_later * weight[:,1:], axis=0),
                      np.sum(price_before * weight[:,0:-1], axis=0)))

# Tracking Error
TE = (1.0/weight.shape[1])*(np.sqrt(np.sum(np.square(rt - Rt))))

# Excess Return
# 提取权重
ER = np.sum(np.mean(
    np.log(np.sum(np.divide(price_later,price_before) * weight[:,1:], axis=0))
    - Rt
))
# 两个权重
# ER = np.sum(np.mean(
#     np.log(np.divide(
#         np.sum(price_later * weight[:,1:], axis=0),
#         np.sum(price_before * weight[:,0:-1], axis=0)))
#     - Rt))

# Sharpe Ratio
mean = np.mean(rt)
var = np.var(rt)
sharpe_ratio = mean / np.sqrt(var)

# Information Ratio
IR = np.divide(ER, TE)

# Tracking Ratio
# tracking_ratio = np.divide(
#             np.divide(np.sum(all_market_capitalization, axis=0),
#                       np.sum(all_market_capitalization[:,0],axis=0)),
#             np.divide(np.sum(weight * market_capitalization, axis=0),
#                       np.sum(weight*np.expand_dims(market_capitalization[:,0],axis=1),axis=0))
#         )

print('TE:',TE)
print('ER:',ER)
print('sharpe ratio:',sharpe_ratio)
print('Information Ratio:',IR)
# print('tracking ratio:',tracking_ratio)

file = open('choose_stock_measure.txt', mode='a')
file.write(str(TE) + ',')
file.write(str(ER) + ',')
file.write(str(sharpe_ratio) + ',')
file.write(str(IR) + ';' + '\n')
file.close()