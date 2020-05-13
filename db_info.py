'''
Created on Jun 29, 2019

create and instance of pg_pandas.PgPandas using a csv file with postgres config info 
    

@author: bperlman1
'''
import pandas as pd
from cme_open_interest import pg_pandas as pg

def get_db_info(config_name=None,db_info_csv_path=None):
    '''
    create an instance of pg_pandas.PgPandas
    :param config_name (default = local:
    :param db_info_csv_path: (default = postgres_info.csv)
    '''
    cf='local' if config_name is None else config_name#digocnjrtr
    dbcsv_path = './postgres_info.csv' if db_info_csv_path is None else db_info_csv_path
    df_dbinfo = pd.read_csv(dbcsv_path)
    s_info = df_dbinfo[df_dbinfo.config_name==cf].iloc[0]
    is_missing = lambda s: str(s).lower() in ['none','nan']
    u = '' if is_missing(s_info.username) else s_info.username
    p = '' if is_missing(s_info.password) else s_info.password
    d = '' if is_missing(s_info.databasename) else s_info.databasename
    print(u,p,d)
    pga = pg.PgPandas(dburl=s_info.dburl,databasename=d,username=u,password=p)    
    return pga

def get_db_info_csv(config_name=None,db_info_csv_path=None):
    '''
    return the config line for config_name
    :param config_name (default = local:
    :param db_info_csv_path: (default = postgres_info.csv)
    '''
    cf='local' if config_name is None else config_name
    dbcsv_path = './postgres_info.csv' if db_info_csv_path is None else db_info_csv_path
    df_dbinfo = pd.read_csv(dbcsv_path)
    s_info = df_dbinfo[df_dbinfo.config_name==cf].iloc[0]
    return s_info
    