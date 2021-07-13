import os, sys
import pymysql
import json
import pandas as pd
from sqlalchemy import create_engine

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
print(path)

settings_file = path + os.sep + "src" + os.sep + "utils" + os.sep + "bd_info.json"

def read_json(fullpath):
    with open(fullpath, "r") as json_file_readed:
        json_readed = json.load(json_file_readed)
    return json_readed

#json = read_json(fullpath=settings_file)

def upload_sql(route, json_readed): # upload_sql(route, json_readed, MySQL)
    # Read the df that we want to upload
    df = pd.read_csv(route)

    # Asign DDBB settings 
    IP_DNS = json_readed["IP_DNS"]
    PORT = json_readed["PORT"]
    USER = json_readed["USER"]
    PASSWORD = json_readed["PASSWORD"]
    BD_NAME = json_readed["BD_NAME"]

    engine = create_engine('mysql+pymysql://' + json_readed['USER'] +
                     ':' + json_readed['PASSWORD'] + '@' + json_readed['IP_DNS'] 
                     + ':' + str(json_readed['PORT'])+ '/' + json_readed['BD_NAME'])

    df.to_sql('marina_serrano_carot', con = engine, if_exists = 'replace', index=False)


class MySQL:

    def __init__(self, IP_DNS, USER, PASSWORD, BD_NAME, PORT):
        self.IP_DNS = IP_DNS
        self.USER = USER
        self.PASSWORD = PASSWORD
        self.BD_NAME = BD_NAME
        self.PORT = PORT
        self.SQL_ALCHEMY = 'mysql+pymysql://' + self.USER + ':' + self.PASSWORD + '@' + self.IP_DNS + ':' + str(self.PORT) + '/' + self.BD_NAME

    
    def connect(self):
        # Open database connection
        self.db = pymysql.connect(host=self.IP_DNS,
                                user=self.USER, 
                                password=self.PASSWORD, 
                                database=self.BD_NAME, 
                                port=self.PORT)
        # prepare a cursor object using cursor() method
        self.cursor = self.db.cursor()
        print("Connected to MySQL server [" + self.BD_NAME + "]")
        return self.db

    def close(self):
        # disconnect from server
        self.db.close()
        print("Close connection with MySQL server [" + self.BD_NAME + "]")
    
    def execute_interactive_sql(self, sql, delete=False):
        """ NO SELECT """
        result = 0
        try:
            # Execute the SQL command
            self.cursor.execute(sql)
            # Commit your changes in the database
            self.db.commit()
            print("Executed \n\n" + str(sql) + "\n\n successfully")
            result = 1
        except Exception as error:
            print(error)
            # Rollback in case there is any error
            self.db.rollback()
        return result
        
    def execute_get_sql(self, sql):
        """SELECT"""
        results = None
        print("Executing:\n", sql)
        try:
            # Execute the SQL command
            self.cursor.execute(sql)
            # Fetch all the rows in a list of lists.
            results = self.cursor.fetchall()
        except Exception as error:
            print(error)
            print ("Error: unable to fetch data")
        
        return results