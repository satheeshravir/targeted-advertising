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
            print line
            columns = line.strip().split(",")
            user_details = { "device_id" : columns[1] , "first_login": columns[2], 
            "mall_first_login": columns[3], "access_point_first": columns[4], "last_login": columns[5],
            "created_at": datetime.datetime.now(), "mall_last_login": columns[6],
            "access_point_last": columns[7]}
            my_data = handle.insert_one(user_details) 
            print my_data

 
    def calculateHistogram(self, imgPath="/home/maxsteal/Pictures/iu.jpg"): 
        gray_img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE) 
        hist = cv2.calcHist([gray_img],[0],None,[256],[0,256]) 
        return hist.ravel().tolist() 
 
    def createUserImageModel(self, user_id,img_path): 
        my_sample_record = { "user_id" : user_id , "histogram": self.calculateHistogram(img_path) 
, "img_path" : img_path, "created_at": 
datetime.datetime.now(),"day":"$day","week":"$week","year":"$year"} 
        handle = self.dbHandle()
        my_data = handle.insert_one(my_sample_record) 
   
    def retrieveUserBasedOnID(self, user_id): 
        handle = self.dbHandle() 
        return handle.find({'user_id':user_id}) 
   
    #Might have to account for time difference 
    def retrieveCurrentWeekHistogram(self, user_id): 
        year, week, dow = datetime.date.isocalendar(datetime.datetime.now()) 
        hist = [] 
        for doc in handle.find({'user_id':user_id,'week':week,'year':year}): 
            hist+=doc['histogram'] 
        return hist 
   
    #Might have to account for time difference 
    def retrieveTodaysMedian(self,user_id): 
        year, week, dow = datetime.date.isocalendar(datetime.datetime.now()) 
        today = time.localtime().tm_yday 
        for doc in handle.find({'user_id':user_id,'day':day,'year':year}): 
            hist+=doc['histogram'] 
        return np.median(hist) 
 
    def aggregateHistograms(self, docs): 
        hist = {} 
        for doc in docs: 
            if doc.user_id in hist: 
                hist[doc.user_id]+=append(doc.histogram) 
            else: 
                hist[doc.user_id] = doc.histogram 
 
        return hist 
   
    def compareHistogram(self, user_id, n): 
        targetUser = self.aggregateHistograms(self.retrieveUserBasedOnID(user_id)) 
        users = self.aggregateHistograms(self.dbHandle().find()) 
        del users[targetUser["user_id"]] 
        topN = [] 
        #Might have to check for minimum number of values to be present in both the distributions 
        for user in users: 
            distance = cv2.compareHist(targetUser['histogram'], user['histogram'], 
CV_COMP_CHISQR) 
            if len(topN) < n+1: 
                topN.append(user['user_id'])             
            else: 
                topN.pop() 
                topN.append(user['user_id']) 
            topN.sort() 
        return topN 



obj = WiFiDataExtraction("mongodb://mongodb0.example.net:27019","quant","customers")
obj.extractCustomerCSV("/home/maxsteal/Downloads/customers.csv")