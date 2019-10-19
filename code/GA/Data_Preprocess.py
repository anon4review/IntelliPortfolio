import xlrd
import pandas as pd
import math
from sklearn import preprocessing
import numpy as np

def read_excel_industry_division(path1):
    #industry division
    financial_list=[]
    governance_list=[]
    infrastructure_list=[]
    resource_list=[]
    transport_list=[]
    growing_list=[]
    value_list=[]
    energy_list=[]
    material_list=[]
    industry_list=[]
    consumption_list=[]
    medicine_list=[]
    goldenlan_list=[]
    information_list=[]
    telecommunication_list=[]
    public_list=[]
    optional_list=[]
    industry_dic={"金融":financial_list,"治理":governance_list,"基建":infrastructure_list,"资源":resource_list,"运输":transport_list,"成长":growing_list,"价值":value_list,
                  "能源":energy_list,"材料":material_list,"工业":industry_list,"消费":consumption_list,"医药":medicine_list,"金地":goldenlan_list,"信息":information_list,
                  "电信":telecommunication_list,"公用":public_list,"可选":optional_list}
    excel = xlrd.open_workbook(path1)
    sheet = excel.sheet_by_index(0)
    for i in range(1,sheet.nrows):
        row_list=sheet.row_values(i)
        for industry in industry_dic.keys():
            if industry in row_list:
                industry_dic[industry].append(row_list[0])
    #remove the company in industry_dic but not in companylist
    companylist = list(pd.read_excel('list.xls')['code'])
    industry_dic_new={}
    for industry in industry_dic:
        #print(industry)
        list_new=[]
        list1 = industry_dic[industry]
        for company in list1:
            if company  in companylist:
                list_new.append(company)
        industry_dic_new[industry] = list_new
    print('行业划分完成')
    return industry_dic_new,companylist

def company_division(path):
    # excel division by company
    data = pd.read_csv(path)
    rows = data.shape[0]
    companylist = []
    for i in range(1,rows):
        temp = data['code'][i]
        if temp not in companylist:
            companylist.append(temp)
    for company in companylist:
        new_df = pd.DataFrame()
        for i in range(0,rows):
            if data['code'][i] == company:
                new_df = pd.concat([new_df,data.iloc[[i],:]],axis = 0, ignore_index = True)
        #caculate the return rate of the stock
        data3 = new_df['closeprice']
        return_rate = []
        return_rate.append(0)
        for i in range(1, data3.shape[0]):
            return_rate.append(math.log(data3.loc[i] / data3.loc[i - 1]))
        data4 = pd.DataFrame(return_rate)
        new_df['ReturnRate'] = data4

        new_df1 = new_df.iloc[-120:-60,:]
        new_df1.to_excel(str(company)+"_train.xls",sheet_name=company,index=False)
        new_df2 = new_df.iloc[-60:, :]
        new_df2.to_excel(str(company)+ "_test.xls", sheet_name=company, index=False)
        print(company)
    new_df = pd.DataFrame(companylist)
    new_df.to_excel('list.xls',index = False)
    return companylist

def caculate_index_return_rate(path):
    #caculate the benchmark index's return rate
    data = pd.read_excel(path, header = 0)
    return_rate=[]
    return_rate.append(0)
    for i in range(1,data.shape[0]):
        return_rate.append(math.log(data.loc[i,'closeprice']/data.loc[i-1,'closeprice']))
    data1 = pd.DataFrame(return_rate)
    data['ReturnRate'] = data1
    data_train = data.iloc[0:60,:]
    data_test = data.iloc[60:120,:]
    data_train.to_excel("index_train.xls")
    data_test.to_excel('index_test.xls')
    return 0

#caculate_index_return_rate('H:\组合投资优化（研究）\上证120天数据\指数行情序列.xlsx')
#company_division('H:\组合投资优化（研究）\上证10年数据\\0918_final.csv')
#print(read_excel_industry_division('H:\组合投资优化（研究）\上证180日度数据（2018）\上证180日度数据（2018）\上证180行业划分.xlsx'))
#read_excel_industry_division('H:\组合投资优化（研究）\上证120天数据\上证180行业划分.xlsx')
