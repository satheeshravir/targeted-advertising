__author__ = "Farhan Khwaja"

import operator
from datetime import datetime
from pymongo import *
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import json


class DataAnalysis:
    def __init__(self, db_path, db_name, collection_name):
        self.db_path = db_path
        self.db_name = db_name
        self.collection_name = collection_name
        self.my_client = None

    def dbHandle(self):
        if self.my_client is None:
            self.my_client = MongoClient(self.db_path)
        my_db = self.my_client[self.db_name]
        collection = my_db[self.collection_name]
        return collection

    def visitsPerDay(self):
        handle = self.dbHandle()
        yValues = []
        xValues = []
        visitedDates = {}
        idTerm = ""
        for record in handle.find():
            dateVal = datetime.strptime(record['date'], '%d/%m/%y')
            timeVal = int(record['time'].split(':')[0])

            if timeVal >= 11 and dateVal.weekday() in [3, 4, 5]:
                visitedDates.setdefault(dateVal, 0)
                visitedDates[dateVal] += 1
            elif timeVal >= 11 and timeVal <= 22:
                visitedDates.setdefault(dateVal, 0)
                visitedDates[dateVal] += 1

        with open('data/VisitsPerDayDist.json', 'w') as outfile:
            json.dump(visitedDates, outfile, indent=4, sort_keys=True)

        print('New File data/VisitsPerDayDist.json saved')


    def visitsPerSeason(self):
        handle = self.dbHandle()
        yValues = []
        xValues = []
        visitedDates = {}
        seasons = {}
        
        for record in handle.find():
            dateVal = datetime.strptime(record['date'], '%d/%m/%y')
            timeVal = int(record['time'].split(':')[0])
            month = dateVal.month
            day = dateVal.day
            if timeVal >= 11 and dateVal.weekday() in [3, 4, 5] and timeVal <= 23:
                if (3, 21) < (month, day) < (6, 20):
                    visitedAP[record['ap_id']].setdefault('SPRING', 0)
                    visitedAP[record['ap_id']]['SPRING'] += 1
                elif (6, 21) < (month, day) < (9, 22):
                    visitedAP[record['ap_id']].setdefault('SUMMER', 0)
                    visitedAP[record['ap_id']]['SUMMER'] += 1
                elif (9, 23) < (month, day) < (12, 21):
                    visitedAP[record['ap_id']].setdefault('FALL', 0)
                    visitedAP[record['ap_id']]['FALL'] += 1
                elif ((12, 21) < (month, day) < (12, 31)) or ((1, 1) < (month, day) < (3, 20)):
                    visitedAP[record['ap_id']].setdefault('WINTER', 0)
                    visitedAP[record['ap_id']]['WINTER'] += 1

            elif timeVal >= 11 and timeVal <= 22:
                if (3, 21) < (month, day) < (6, 20):
                    visitedAP[record['ap_id']].setdefault('SPRING', 0)
                    visitedAP[record['ap_id']]['SPRING'] += 1
                elif (6, 21) < (month, day) < (9, 22):
                    visitedAP[record['ap_id']].setdefault('SUMMER', 0)
                    visitedAP[record['ap_id']]['SUMMER'] += 1
                elif (9, 23) < (month, day) < (12, 21):
                    visitedAP[record['ap_id']].setdefault('FALL', 0)
                    visitedAP[record['ap_id']]['FALL'] += 1
                elif ((12, 21) < (month, day) < (12, 31)) or ((1, 1) < (month, day) < (3, 20)):
                    visitedAP[record['ap_id']].setdefault('WINTER', 0)
                    visitedAP[record['ap_id']]['WINTER'] += 1

        with open('data/VisitsPerSeasonDist.json', 'w') as outfile:
            json.dump(visitedAP, outfile, indent=4, sort_keys=True)

        print('New File data/VisitsPerSeasonDist.json saved')

    def accessPointPerDay(self):
        handle = self.dbHandle()
        yValues = []
        xValues = []
        visitedAP = {}
        for record in handle.find():
            visitedAP.setdefault(record['ap_id'], {})
            dateVal = datetime.strptime(record['date'], '%d/%m/%y')
            timeVal = int(record['time'].split(':')[0])

            if timeVal >= 11 and dateVal.weekday() in [3, 4, 5]:
                visitedAP[record['ap_id']].setdefault(dateVal, 0)
                visitedAP[record['ap_id']][dateVal] += 1
            elif timeVal >= 11 and timeVal <= 22:
                visitedAP[record['ap_id']].setdefault(dateVal, 0)
                visitedAP[record['ap_id']][dateVal] += 1

        with open('data/AccessPointPerDayDistribution.json', 'w') as outfile:
            json.dump(visitedAP, outfile, indent=4, sort_keys=True)

        print('New File data/AccessPointPerDayDistribution.json saved')

        #UnComment Below to plot the chart

        # f, axarr = plt.subplots(len(visitedAP.keys()), sharex=True)
        # i = 0
        # for key, value in visitedAP.items():
        #     print('AP: ',key)
        #     for dt, val in value.items():
        #         xValues.append(dt.date())
        #         yValues.append(val)

        #     axarr[i].bar(xValues, yValues)
        #     axarr[i].set_title('Access Point: '+key)
        #     i += 1
        # plt.show()


    def accessPointPerSeasonDist(self):
        handle = self.dbHandle()
        yValues = []
        xValues = []
        visitedAP = {}
        for record in handle.find():
            visitedAP.setdefault(record['ap_id'], {})
            dateVal = datetime.strptime(record['date'], '%d/%m/%y')
            month = dateVal.month
            day = dateVal.day
            timeVal = int(record['time'].split(':')[0])

            if timeVal >= 11 and dateVal.weekday() in [3, 4, 5] and timeVal <= 23:
                if (3, 21) < (month, day) < (6, 20):
                    visitedAP[record['ap_id']].setdefault('SPRING', 0)
                    visitedAP[record['ap_id']]['SPRING'] += 1
                elif (6, 21) < (month, day) < (9, 22):
                    visitedAP[record['ap_id']].setdefault('SUMMER', 0)
                    visitedAP[record['ap_id']]['SUMMER'] += 1
                elif (9, 23) < (month, day) < (12, 21):
                    visitedAP[record['ap_id']].setdefault('FALL', 0)
                    visitedAP[record['ap_id']]['FALL'] += 1
                elif ((12, 21) < (month, day) < (12, 31)) or ((1, 1) < (month, day) < (3, 20)):
                    visitedAP[record['ap_id']].setdefault('WINTER', 0)
                    visitedAP[record['ap_id']]['WINTER'] += 1

            elif timeVal >= 11 and timeVal <= 22:
                if (3, 21) < (month, day) < (6, 20):
                    visitedAP[record['ap_id']].setdefault('SPRING', 0)
                    visitedAP[record['ap_id']]['SPRING'] += 1
                elif (6, 21) < (month, day) < (9, 22):
                    visitedAP[record['ap_id']].setdefault('SUMMER', 0)
                    visitedAP[record['ap_id']]['SUMMER'] += 1
                elif (9, 23) < (month, day) < (12, 21):
                    visitedAP[record['ap_id']].setdefault('FALL', 0)
                    visitedAP[record['ap_id']]['FALL'] += 1
                elif ((12, 21) < (month, day) < (12, 31)) or ((1, 1) < (month, day) < (3, 20)):
                    visitedAP[record['ap_id']].setdefault('WINTER', 0)
                    visitedAP[record['ap_id']]['WINTER'] += 1

        with open('data/AccessPointPerSeasonDistribution.json', 'w') as outfile:
            json.dump(visitedAP, outfile, indent=4, sort_keys=True)

        print('New File data/AccessPointPerSeasonDistribution.json saved')


if __name__ == "__main__":
    obj = DataAnalysis("mongodb://localhost:27017/","quant","logins")
    obj.accessPointPerSeasonDist()
    print("Started!!")