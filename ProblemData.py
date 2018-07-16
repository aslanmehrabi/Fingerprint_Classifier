
import pandas as pd
import json
import csv

from collections import defaultdict
from datetime import datetime
import time
import numpy
import dill as pickle
import random
from sklearn.model_selection import train_test_split


#print ('=============*************')


class ProblemData:
    #numZones = 449
    #numMaxAdds = 19

    defaultSignalVal = 0  # the min amount of received signal (~ zero value)
    numNodes = 0  # number of sensors(node)
    moment = []  # Time scope of received signal from a suser
    signal = []  # Dictionary of received finger prints
    sigMat = 0  # Matrix of received fignerprints (dimensions: #row_input * # nodes )
    macAdd = []  # array of Mac add of mobile phones
    zone = []  # array of zones of users in each line on input
    numRows = 0  # total number of rows of input which should be considered



    def __init__(self, defaultSignalValue = -100, numNodes = 277):
        self.defaultSignalVal = defaultSignalValue
        self.numNodes = numNodes

    def loadData(self,  useStoredData=False, inputFileName='', storeReadData = 'False',storeDataName='default_store_name', rowReadUntil = -1):
        if useStoredData:
            storedData = open(storeDataName, 'rb')
            self = pickle.load(storedData)
            storedData.close()
            return self

        else:
            if (inputFileName == ''):
                print('======> inputFileName field should be passed')
                return self

            self.readInput(fileName=inputFileName,rowReadUntil=rowReadUntil)  #To reaad limited number of row
            if(storeReadData):
                storedData = open(storeDataName, 'wb')
                pickle.dump(self, storedData)
                storedData.close()
            return self


    # read the data from the input file:
    def readInput(self , fileName, rowReadUntil = -1):
        dataAddrMain = fileName
        with open(dataAddrMain) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')


            if rowReadUntil != -1:
                self.numRows = rowReadUntil
            else:
                self.numRows = sum(1 for row in readCSV) - 1
                csvfile.seek(0)  # moving back to the first line of the input

            # next(reader) # skip header
            for idx, row in enumerate(readCSV):
                if idx == 0:
                    header = row
                else:
                    #if(idx % 10000 == 0):
                       #print('row reading: ',idx)
                    tmpTime = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                    self.moment.append(time.mktime(tmpTime.timetuple()))
                    self.signal.append(json.loads(row[1].replace("'", "\"")))  # converting to dict type
                    self.macAdd.append(int(row[2]))
                    self.zone.append(int(row[3].split(' ')[1]))
                    if (idx == self.numRows):  # to read limited number of rows defined by rowReadUntil
                        break

            self.signal = [dict([int(a), int(x)] for a, x in b.items()) for b in self.signal]  # converting signals to int values
            self.signal = [defaultdict(lambda: self.defaultSignalVal, sig) for sig in self.signal]  # add default signal value for probes which did not get any signal
            self.sigMat = numpy.full((self.numRows, self.numNodes), self.defaultSignalVal)  # signal matrix: each node is a column
            for i in range(self.numRows):
                self.sigMat[i][list(self.signal[i].keys())] = list(self.signal[i].values())  # for i in range(self.numRows)

            self.zone = numpy.full(self.numRows, self.zone)  # convering zone to numpy Array
            self.moment = numpy.full(self.numRows, self.moment)
            self.macAdd = numpy.full(self.numRows, self.macAdd)
            return self.checkRecievedData()


    def checkRecievedData(self):  # can be added to check the validity of read input
        return True


