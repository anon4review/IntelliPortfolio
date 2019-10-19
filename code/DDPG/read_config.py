import configparser
config=configparser.ConfigParser()
config.read('./config.ini')

LAMDA = float(config.get('ddpg','lamda'))