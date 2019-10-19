import xlrd
import pandas as pd
import sqlite3
from preprocess.norm_6_features import norm
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
    companylist = list(pd.read_excel('./list.xls')['code'])
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


def clearOrCreate_table(conn,name):
    SQL = 'create table if not exists %s (code string,date date,openprice double,' \
              'maxprice double,minprice double, closeprice double,vol double,money double,updown double,' \
              'updownratio double,meanprice double, TurnoverRate double,CirculationValue double,TotalValue double,' \
              'FlowEquity double,TotalEquity double,PE double,PB double,P2S double,PCF double)'%name

    conn.execute(SQL)
    SQL='delete from %s'%name
    conn.execute(SQL)
    print('table %s create ready!'%name)


class Portfolio(object):
    #given n stocks
    #return l stocks with different weights
    def init(self,l):
        self.l = l
        self.industry_dic,self.companylist = read_excel_industry_division(path1='./上证120天数据/上证180行业划分.xlsx')
        self.mean_mc_list, self.mean_amount_list,self.w_m = self.caculate_amount_mc_scaled()

    def caculate_amount_mc_scaled(self):
        #caculate the mean traing amount and market capitalization of each company
        #scale the list of the mean
        mean_amount_list=[]
        mean_mc_list=[]
        for company in self.companylist:
            data = pd.read_excel('./train_test_data/'+str(company) + "_train.xls")
            mean_amount = data['vol'].mean()
            mean_mc = data['TotalValue'].mean()
            mean_amount_list.append(mean_amount)
            mean_mc_list.append(mean_mc)
        #各个公司市值均值站市场总市值的比例，ga的适应度函数需要用到
        w_m = []
        sum_ = sum(mean_mc_list)
        for mean_mc in mean_mc_list:
            w_m.append(mean_mc/sum_)
        #scale
        max1 = max(mean_amount_list)
        min1 = min(mean_amount_list)
        for element in mean_amount_list:
            mean_mc_list[mean_amount_list.index(element)] = (element-min1)/(max1-min1)
        max2 = max(mean_mc_list)
        min2 = min(mean_mc_list)
        for element in mean_mc_list:
            mean_mc_list[mean_mc_list.index(element)] = (element-min2)/(max2 - min2)
        print('市值、交易量均值计算及归一化完成')
        return mean_mc_list,mean_amount_list,w_m

    def caculate_industry_market_capitalization(self):
        # caculate industry market capitalization
        market_capitalization_list = []
        for industry in self.industry_dic:
            company_list = self.industry_dic[industry]
            market_capitalization = 0
            for company in company_list:
                data = pd.read_excel('./train_test_data/'+str(company) + "_train.xls")
                #使用市值的均值
                mc = data['TotalValue'].mean()
                market_capitalization += mc
            market_capitalization_list.append(market_capitalization)
        print('各行业总市值获取完成')
        return market_capitalization_list

    def capitalization_industry_max(self):
        #return the position of the max
        market_capitalization_list = self.caculate_industry_market_capitalization()
        max_position = market_capitalization_list.index(max(market_capitalization_list))
        print('市值最大行业获取完成')
        return max_position

    def judge_stock_select(self,v1,v2,v3):
        #After having the industry sector having the largest amount of market capitalization,we can select the company in this industry sector
        #select the stock having the highest P_ij to add to the portfolio.
        position = self.capitalization_industry_max()
        part_company_list = list(self.industry_dic.values())[position]
        max_P_ij = 0
        max = part_company_list[0]
        for company in part_company_list:
            data = pd.read_excel('./train_test_data/'+str(company)+'_train.xls')
            data1 = pd.read_excel('index_train.xls')
            A = self.mean_amount_list[self.companylist.index(company)]
            M = self.mean_mc_list[self.companylist.index(company)]
            B = (data['ReturnRate'].drop(index=0).std())/((data1['ReturnRate'].drop(index=0).var()*(data1.shape[0]-1))**0.5)
            P_ij = v1*(1/B)+v2*A+v3*M
            if(P_ij > max_P_ij):
                max = company
        print('最大行业选股完成')
        return max

    def update(self,select_stock):
        # remove the stock selected from the industry sector
        # update the sort of market capitalization by industry
        industry_dic={}
        for industry in self.industry_dic:
            if select_stock in self.industry_dic[industry]:
                list1 = self.industry_dic[industry].copy()
                list1.remove(select_stock)
                industry_dic[industry] = list1
            else:
                industry_dic[industry] = self.industry_dic[industry]
        self.industry_dic = industry_dic
        print('行业字典更新完成')

    def select_l_stocks(self,v1,v2,v3):
        #caculate and select stock,then update the industry_dic until the portfolio have l stocks
        portfolio=[]
        market_capitalization_list = []
        for i in range(self.l):
            select_stock = self.judge_stock_select(v1,v2,v3)
            portfolio.append(select_stock)
            market_capitalization_list.append(self.w_m[self.companylist.index(select_stock)])
            self.update(select_stock)
            print('第'+str(i+1)+'支股票选取完成：'+select_stock)
        print(str(self.l)+'支股票选取完成')
        return portfolio,market_capitalization_list

if __name__=='__main__':
    port = Portfolio()
    port.init(l = 10)
    portfolio,mc_list = port.select_l_stocks(v1=2,v2=1,v3=1)
    print(portfolio)
    conn = sqlite3.connect(database='../data/SH180.db')
    clearOrCreate_table(conn,'industry_choosen_stocks_data')

    for stock in portfolio:
        SQL = 'insert into industry_choosen_stocks_data select * from originData where code = \"%s\"'  % stock
        conn.execute(SQL)
    norm(conn,'industry_choosen_stocks_data','industry_choosen_stocks_norm')

    print('completed')



