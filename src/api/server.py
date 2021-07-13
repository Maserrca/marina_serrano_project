'''
Contains the functionality that starts the Flask API. 
This file only will be executed if it is passed an argument“-x 8642” (argparse), 
otherwisewill show “wrong password”. In the API there is, atleast, this GET functions:a.
One that must allow you to receive atoken_idvalueand, iftoken_idis equaltoS, return the jsons that contains the logic explainedbelow. 
Otherwise,return a string with a message of error.i.Sis the DNI of the student starting with the letter: Example:“B80070012”.ii.
The json returned by flask must be the cleaned dataof your data.8.It is mandatory that the code runs on any computerwith no errors (relative paths)
'''
# ------- Import the necessary libraries -------
from flask import Flask, request, render_template
import os,sys
import argparse
from sqlalchemy import create_engine
import json


# Access to the folder and append the path
dir = os.path.dirname
path = dir(dir(dir(os.path.abspath(__file__))))
sys.path.append(path)
print(path)
# ------ Import functions from apis_tb ------
from src.utils.apis_tb import *
from src.utils.sql_tb import *
# ------ Create Flask ------
app = Flask(__name__)

@app.route("/") 
def home():
    """ Default path """
    return "Do you have a tumor?"

@app.route('/give_me_id', methods=['GET'])
def give_id():
    x = request.args['token_id']
    if x == "M53994161": 
        return return_json(path+ os.sep + "resources" + os.sep + "photos.json")
    else:
        return "Wrong password"

@app.route('/sql', methods=['GET'])
def sql():
    x = request.args['upload']
    if x == 'yes':
        dfpath = path + os.sep + 'data' + os.sep + 'df_photos.csv'
        json_readed = read_json(fullpath=settings_file)
        upload_sql(dfpath, json_readed)
        
        return "Eureka"
    else:
        return "Damn it"


def main():
    
    settings_file = path +  os.sep + 'resources' + os.sep + "json.json"
    
    # Load json from file
    json_readed = read_json(fullpath= settings_file)

    DEBUG = json_readed["debug"]
    HOST = json_readed["host"]
    PORT_NUM = json_readed["port"] 

    app.run(debug=DEBUG, host=HOST, port=PORT_NUM)

parser = argparse.ArgumentParser()
parser.add_argument("-x", "--x", type=str, help="password")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    print(args.values())
    if args["x"] == "marina": 
        main()
    else:
        print("wrong password")
