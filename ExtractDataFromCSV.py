import datetime 
from pymongo import * 
import numpy as np 
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
 
 
class WiFiDataExtraction: 
    def __init__(self, db_path, db_name, collection_name): 
        self.db_path = db_path 
        self.db_name = db_name 
        self.collection_name = collection_name 
        self.my_client = None 
   
    def dbHandle(self): 
        if self.my_client == None: 
            self.my_client = MongoClient(self.db_path) 
        my_db = self.my_client[self.db_name] 
        collection = my_db[self.collection_name] 
        return collection 

    def extractCustomerCSV(self, filePath):
        readFile = open(filePath, "r")
        readFile.readline()
        handle = self.dbHandle()         
        histValues = []
        logins = {}
        idTerm = ""
        for line in readFile:
            columns = line.strip().split(",")
            user_details = { "dev_id": columns[2], "account_id": columns[0], 
            "ap_id": columns[1], "date": columns[3].split()[0], "time":columns[3].split()[1],
            "created_at": datetime.datetime.now()}
            if idTerm == "":
                idTerm = columns[2]
                logins[columns[2]] = {}
                key = user_details["date"]+user_details["time"].split(":")[0]
                logins[columns[2]][key] = user_details
            elif idTerm != "" and columns[2] in logins:
                key = user_details["date"]+user_details["time"].split(":")[0]
                logins[columns[2]][key] = user_details

            else:
                for key, value in logins.items():
                    for key, item in value.items():
                        histValues.append(int(item["time"].split(":")[0]))
                        my_data = handle.insert_one(item)                     
                idTerm = ""
                logins = {}
        plt.hist(histValues)
        plt.show()


            
        #print my_data



obj = WiFiDataExtraction("mongodb://localhost:27017/","quant","logins")
obj.extractCustomerCSV("/home/maxsteal/Downloads/logins.csv")