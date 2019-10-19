import configparser

config=configparser.ConfigParser()
config.read('./config.ini')

ALL_DAY_NUM=int(config.get('data', 'all_days_used'))
STOCK_NUM=int(config.get('data', 'stock_num'))
CHOOSEN_STOCK_NUM=int(config.get('data', 'stock_choose_num'))
FEATURE_NUM=int(config.get('data', 'feature_num'))
TRAIN_NUM=int(config.get('data', 'train_num'))
VAL_NUM=int(config.get('data', 'val_num'))
TEST_NUM=int(config.get('data', 'test_num'))

WIN_LEN=int(config.get('rl', 'win_len'))
BATCH_SIZE=int(config.get('rl', 'batch_size'))
DECISION_DAY_NUM=int(config.get('rl', 'decision_day_num'))
LR=float(config.get('rl', 'lr'))
MAX_EPOCH=int(config.get('rl', 'max_epoch'))
# IF_CONSIDER_INDEX=config.get('rl', 'if_consider_index')
LAMDA = float(config.get('rl','lamda'))

DB_PATH=str(config.get('db', 'db_path'))
TABLE_NAME=str(config.get('db', 'table_name'))
