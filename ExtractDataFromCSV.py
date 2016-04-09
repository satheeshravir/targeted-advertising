import datetime 
from pymongo import * 
import cv2 
import numpy as np 
 
 
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
        print "test"
        readFile = open(filePath, "r")
        readFile.readline()
        handle = self.dbHandle()         
        for line in readFile:
            columns = line.strip().split(",")
            user_details = { "device_id" : columns[1] , "first_login": columns[2], 
            "mall_first_login": columns[3], "access_point_first": columns[4], "last_login": columns[5],
            "created_at": datetime.datetime.now(), "mall_last_login": columns[6],
            "access_point_last": columns[7]}
            my_data = handle.insert_one(user_details) 
            print my_data



obj = WiFiDataExtraction("mongodb://localhost:27017/","quant","customers")
obj.extractCustomerCSV("/home/maxsteal/Downloads/customers.csv")