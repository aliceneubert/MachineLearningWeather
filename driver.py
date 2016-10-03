#!/usr/bin/python3

from sklearn import linear_model
import datetime
import json

def mean(l):
    if len(l) == 0:
        return 0 #Hopefully it wouldn't use this column anyway
    return sum(l)/float(len(l))

with open('jsondata.json', 'r') as datafile:
   data = json.loads(datafile.read())

PRCP = [float(line[21]) for i,line in enumerate(data) if i!=0 and line[21] != '-9999'] #Get all the precipitation data, minus its label

labels = data[0]
data = data[1:] #remove labels
data = [[float(col) for i,col in enumerate(line) if i!=21 and i >=2] for line in data if line[21] != "-9999"] #get the precipitation data out of the other data

averages = [mean([line[i] for line in data if line[i]!=-9999]) for i,col in enumerate(data[0])]

for i,line in enumerate(data):
    for j,feature in enumerate(line):
        if feature == -9999:
            data[i][j] = averages[j]

reg = linear_model.Ridge(alpha=.5)
print(reg.fit(data, PRCP))
print(reg.coef_)
print(reg.intercept_)

today = float(datetime.datetime.strftime(datetime.datetime.now(), "%Y%M%d"))

print("Today: ",today)

mostRecent = data[-1]
guessData = mostRecent
guessData[3] = today

guess = sum([g*c for g,c in zip(guessData,reg.coef_)]) + reg.intercept_
print(guess)

def getPrediction(guessData):
    return sum([g*c for g,c in zip(guessData,reg.coef_)]) + reg.intercept_

recentData = data[-9000:]
recentPRCP = PRCP[-9000:]
for i,datum in enumerate(recentData):
    if PRCP[i] > .2:
        print("Actual: " + str(PRCP[i]))
        print("Predicted: " + str(getPrediction(datum)))
        print("")
