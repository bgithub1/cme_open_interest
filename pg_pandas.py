'''
Created on Aug 22, 2016

@author: bperlman1
general access to postgres via pandas and sqlalchemy
''' 

import pandas as pd
from sqlalchemy import create_engine
# import sys
# import datetime 
# import psycopg2 as ps
import os
import sqlalchemy as sa
from sqlalchemy.sql.expression import bindparam
import sqlalchemy.orm as orm
# import pandasql as psql
# import json
# import re
from io import StringIO as sio
import inspect
import zipfile 
DB_CSV_DEFAULT_PATH = "./db.csv"

import logging

def init_root_logger(logfile,logging_level=None):
    level = logging_level
    if level is None:
        level = logging.DEBUG
    # get root level logger
    logger = logging.getLogger()
    if len(logger.handlers)>0:
        return logger
    logger.setLevel(logging.getLevelName(level))

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)   
    return logger

class PgPandas(object):
    '''
    PgPandas facilitates Pandas access to a Postgres database 
    '''
    def __init__(self,
                 username=None,
                 password=None,
                 dburl=None,
                 databasename=None,
                 dbflavor=None,
                 engine=None,
                 logger=None):
        if logger is None:
            self.logger = init_root_logger("logfile.log","INFO")
        else:
            self.logger = logger
            
        self.dbflavor = dbflavor if dbflavor is not None else 'postgresql://'
        
        if engine is None:
            if username is None or password is None or dburl is None or databasename is None:
                missing_list = []
                if username is None:
                    missing_list.append("username")
                if password is None:
                    missing_list.append("password")
                if dburl is None:
                    missing_list.append("dburl")
                if databasename is None:
                    missing_list.append("databasename")
                s = ','.join(missing_list)
                raise ValueError("%s is missing from PgPandas constructor" %(s))
            self.username=username
            self.password=password
            self.dburl=dburl
            self.databasename=databasename
            self.engine = self._get_engine()
        else:
            self.engine = engine
        pass

    
    def _get_engine(self,db_flavor=None):
        engine_string = self.dbflavor + self.username + ':'+ self.password + '@'
        engine_string += self.dburl + '/' + self.databasename
        engine = create_engine(engine_string)
        return engine

    def put_df_fast(self,df,table_name_with_schema):
        col_names = df.columns.values
        unnest_list = []
        for c in col_names:
            unnest_list.append("unnest(array%s)" %(list(df[c])))
        all_unnests = ','.join(unnest_list)
        col_name_string = ','.join(col_names)
        sql='''INSERT INTO %(TABLE_NAME)s(%(COL_NAMES)s)
               SELECT %(UNNEST_LIST)s
        ''' %{'TABLE_NAME':table_name_with_schema,'COL_NAMES':col_name_string,'UNNEST_LIST':all_unnests}
        self.exec_sql_raw(sql)
        pass


    def exec_sql_raw(self,sql):
        '''
            Called postgres sql statement when the sql references functions that are
            not in the same schema as the login referenced by db_csv_path
        '''
        cur = self.engine.connect()
        result = cur.execute(sql)
        cur.close()
        return result

    def exec_stored_procedure(self,function_name,function_arg_list):
        '''
            Called postgres stored procedure
        '''
        cur = self.engine.connect()
        cur.callproc(function_name,function_arg_list)
        cur.close()


    def get_sql(self,sql_string):
        ret = pd.read_sql_query(sql_string,con=self.engine)
        return ret
    
    def get_sqlfile(self,sql_file_path):
        with open(sql_file_path, 'r') as myfile:
            sql_string=myfile.read()
            return self.get_sql(sql_string)

    def write_df_to_postgres_using_metadata(self,df,table_name):
        meta = sa.MetaData(bind=self.engine)
        table_name_only = table_name.split(".")[1]
        schema_only = table_name.split(".")[0]
        docs = sa.Table(table_name_only,meta,autoload=True,schema=schema_only)
        sess = orm.sessionmaker(bind=self.engine)()
        conn = self.engine.connect()
        listToWrite = df.to_dict(orient='records')
        conn.execute(docs.insert(),listToWrite)
        sess.commit()
        sess.close()
        
    
    def update_df_to_postgres_using_metadata_and_id(self,df,table_name):
        meta = sa.MetaData(bind=self.engine)
        table_name_only = table_name.split(".")[1]
        schema_only = table_name.split(".")[0]
        docs = sa.Table(table_name_only,meta,autoload=True,schema=schema_only)
        sess = orm.sessionmaker(bind=self.engine)()
        conn = self.engine.connect()
        df_2 = df.copy()
        df_2['_id'] = df_2['id']
        value_columns = filter(lambda c: c != 'id',df.columns.values)
        df_2 = df_2[['_id']+value_columns]
        listToWrite = df_2.to_dict(orient='records')
        value_dict = {}
        for vc in value_columns:
            value_dict[vc] = bindparam(vc)
        conn.execute(docs.update().where(docs.c.id == bindparam('_id')).values(value_dict),listToWrite)
        sess.commit()
        sess.close()
    

    def df_to_excel(self,df_list,xlsx_path,sheet_name_list=None):
        writer = pd.ExcelWriter(xlsx_path)
        sn_list = sheet_name_list
        if sn_list is None:
            ''' create it '''
            num_list = range(1,(len(df_list)+1))
            sn_list = ['Sheet' + str(x) for x in num_list]
        for i in range(0,len(df_list)):
            df_list[i].to_excel(writer,sn_list[i])
        writer.save()  

    def write_binary_data_to_file(self,document_binary,output_file_path):
        s = str(document_binary)    
        with open(output_file_path, 'w') as myfile:
            myfile.write(s)
    

    def write_binary_data_to_file_with_sql(
            self,
            sql,
            blob_field_name,
            file_name_field_name,
            output_folder,
            file_name_adaptor=None):
        """
        Write a group of binary files to an output folder, given an sql statements
        :param engine:
        :param sql:
        :param blob_field_name:
        :param file_name_field_name:
        :param output_folder:
        :param file_name_adaptor: a python method that will take a row of the input dataframe,  and convert it to something that makes it usable to write to a local folder
        """
        df_doc_bin = self.get_sql(sql)
        if df_doc_bin is None or len(df_doc_bin)<=0:
            return None
        if  blob_field_name not in df_doc_bin.columns.values:
            return None
        if  file_name_field_name not in df_doc_bin.columns.values:
            return None
        full_path = []
        
        for i in range(len(df_doc_bin)):
            name = str(df_doc_bin.iloc[i][file_name_field_name])
            if file_name_adaptor is not None:
                name  = file_name_adaptor(df_doc_bin.iloc[i])
            output_file_path = output_folder+"/" + name 
            print ("write_binary_data_to_file_with_sql: writing data to " + output_file_path)   
            s = str(df_doc_bin.iloc[i][blob_field_name])
            with open(output_file_path, 'w') as myfile:
                    myfile.write(s)
            full_path.append(output_file_path)
        df_full_path = pd.DataFrame({'full_path':full_path})
        df_return = pd.concat([df_doc_bin,df_full_path],axis=1)
        return df_return
    
    def write_binary_data_to_zip_file_with_sql(
            self,
            sql,
            blob_field_name,
            file_name_field_name,
            output_file,
            file_name_adaptor=None):
        """
        Write a group of binary files to an output folder, given an sql statements
        :param engine:
        :param sql:
        :param blob_field_name:
        :param file_name_field_name:
        :param output_file: this an actual file object (could be a memory_file, as in memory_file = io.BytesIO()
        :param file_name_adaptor: a python method that will take a file name, and convert it to something that makes it usable to write to a local folder
        """
        df_doc_bin = self.get_sql(sql)
        if df_doc_bin is None or len(df_doc_bin)<=0:
            return None
        if  blob_field_name not in df_doc_bin.columns.values:
            return None
        if  file_name_field_name not in df_doc_bin.columns.values:
            return None
        dict_files = {}
        for i in range(len(df_doc_bin)):
            name = str(df_doc_bin.iloc[i][file_name_field_name])
            if file_name_adaptor is not None:
                name  = file_name_adaptor(name,df_doc_bin.iloc[i])
            print ("write_binary_data_to_file_with_sql: writing data to " + name)   
            s = str(df_doc_bin.iloc[i][blob_field_name])
            dict_files[name] = s
        # now make zip file
        with zipfile.ZipFile(output_file, 'w') as zf:
            for k in dict_files.keys():
                individual_file_name = k 
                individual_file_binary = dict_files[k]
                data = zipfile.ZipInfo(individual_file_name)
                data.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(data, individual_file_binary)
        return 

def make_zip_file(file_folder_path,file_name_list,output_file):
    """
    Turn output_file into a zip file.  
    
    :param file_folder_path: the full path to the location of all of the files in file_name_list
    :param file_name_list: list of file_names, without paths
    :param output_file: file object (NOT A FILE NAME or PATH) which will hold the zipped contents
    """
    ffp = file_folder_path
    if ffp is None:
        ffp = ""
    else:
        ffp += "/"
    with zipfile.ZipFile(output_file, 'w') as zf:
        for file_name in file_name_list:
            fpath = ffp + str(file_name)
            if not os.path.isfile(fpath):
                continue
            file_data = open(fpath,'r').read() 
            data = zipfile.ZipInfo(file_name)
            data.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(data, file_data)
    
def write_document_binary_to_text_file(document_binary,output_file_path):
    open(output_file_path,'w').writelines(str(document_binary).split('\\n'))

        
    
    
def get_full_path_of_import(import_module_reference):
    """
    GET THE FOLDER THAT HOLDS THE MODULE import_module_reference
    Example: (see bankprocessing qba)
    import machine.mach
    import pypg.pg_pandas as pg
    folder_of_mach = pg.get_full_path_of_import(pg)
    
    :param import_module_reference: a reference (NOT THE STRING REFERENCE) to an imported module
    """
    f = inspect.getfile(import_module_reference)
    p = os.path.split(f)
    return p[0]
#     path_split = inspect.getfile(import_module_reference).split("/")
#     ret_path = "/".join(path_split[:len(path_split)-1])
#     return ret_path

def print_source(import_module_method_reference):
    """
    Examples (this example can be found in machine/machine/test_qba.py: 

    import mach
    import pypg.pg_pandas as pg
    if __name__=='__main__':
        qba = mach.qbafp()
        pg.print_source(mach.qbafp)
        pg.print_source(qba.gfn)
    
    
    :param import_module_method_reference: a reference (NOT THE STRING REFERENCE) to a method in an imported module
    """
    lines = inspect.getsource(import_module_method_reference).split('\n')
    for l in lines:
        print (l)   

def df_print(df):    
    """
    Print all rows of a dataframe
    :param df:
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        print(df)

def df_to_string(df):    
    """
    Print all rows of a dataframe
    :param df:
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        return df.to_string()

def df_find_header(df,head_value_to_find,col_num_to_search=None):
    """
    EXPERIMENTAL
    """
    if df.columns.values[col_num_to_search]==head_value_to_find:
        return df
    actual_header_row = -1
    df2 = df.copy()
    df2.index = range(len(df))
    for i in range(10):
        if head_value_to_find.lower() in str(df2.ix[i,col_num_to_search]).lower():
            actual_header_row = i 
            break
    if actual_header_row<0:
        return None
    
    s = sio.StringIO()
    text_lines = df2.to_string().split('\n')
    text_lines = text_lines[actual_header_row+1:len(text_lines)]
    s = sio.StringIO()
    s.writelines(text_lines)
    df_ret = pd.read_csv(s,dtype=str)
    return df_ret
        
def filter_import(import_module_reference,names_to_search_for):
    """
    This is an easy way to avoid typing "filter(lambda s: names_to_search_for in s,dir(import_module_reference))" at the console
    :param import_module_reference: a reference to an import like pg, or mach (NOT quoted, like "pg" or "mach")
    :param names_to_search_for: the whole or partial name of a method or variable in the import
    """
    return filter(lambda s: names_to_search_for in s,dir(import_module_reference))

def ls_methods(import_module_reference):
    module_names = [func for func in dir(import_module_reference) if 'function' in str(type(getattr(import_module_reference,func)))]
    module_args = [inspect.getargspec(getattr(import_module_reference, mn)) for mn in module_names]
    df_ret = pd.DataFrame({'module':module_names,'module_args':module_args})
    return df_ret

def pd_widen():
    pd.set_option('display.width', 200)
        




    
# class ends here

