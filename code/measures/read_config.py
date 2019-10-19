import configparser

config=configparser.ConfigParser()
config.read('./config.ini')
DB_PATH=str(config.get('db', 'db_path'))
BEGIN_DATE=int(config.get('date', 'begin_date'))
END_DATE=int(config.get('date', 'end_date'))
WIN_LEN=int(config.get('data', 'win_len'))
CHOOSEN_STOCK_NUM=int(config.get('data', 'choosen_stock_num'))