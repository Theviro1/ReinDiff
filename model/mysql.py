import pymysql
import pymysql.cursors
from config.default_config import *

class MySQL:
    def __init__(
        self, 
        host:str=HOST, 
        port:str=PORT, 
        user:str=USER, 
        password:str=PASSWORD, 
        database:str=DATABASE,
        table:str=TABLE
    ):
        self.connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            cursorclass=pymysql.cursors.DictCursor
        )
        self.table = table
    
    def insert(self, image_path:str, num_classes:int, prompt:str, annotation_path:str):
        try:
            with self.connection.cursor() as cursor:
                sql = f'INSERT INTO {self.table} (image_path, num_classes, prompt, annotation_path) values (%s, %s, %s, %s)'
                values = (image_path, num_classes, prompt, annotation_path)
                cursor.execute(sql, values)
            self.connection.commit()
            print(f'commited')
        except:
            print('exception occurs')
            pass
    
    def insert_column(self, id:int, column_name:str, column_data):
        try:
            with self.connection.cursor() as cursor:
                sql = f'UPDATE {self.table} SET {column_name} = {column_data} WHERE id = {id}'
                cursor.execute(sql)
            self.connection.commit()
        except:
            print('exception occurs')
        return
    
    def select_by_index(self, startindex:int, endindex:int):
        try:
            with self.connection.cursor() as cursor:
                sql = f'SELECT * FROM {self.table} ORDER BY id ASC LIMIT {startindex}, {endindex}'
                cursor.execute(sql)
                results = cursor.fetchall()
            return results  # list of dict contain keys 'id', 'image_path', 'num_classes', 'prompt', 'annotation_path'
        except:
            print('exception occurs')
            pass
    
    def select_all(self):
        try:
            with self.connection.cursor() as cursor:
                sql = f'SELECT * FROM {self.table} ORDER BY id ASC'
                cursor.execute(sql)
                results = cursor.fetchall()
            return results
        except:
            print('exception occurs')
            pass
