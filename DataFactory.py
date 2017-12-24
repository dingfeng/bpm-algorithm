# -*- coding: UTF-8 -*-
# filename: DataFactory date: 2017/12/24 11:39  
# author: FD 
from numpy import *

tags = [0, 1]
userIdSets = [range(20), range(20, 40), range(40, 60), range(60, 80), range(80, 100)]
userIdSetIndex = 0
problemIdSets = [range(20), range(20, 40), range(40, 60), range(60, 80), range(80, 100)]
endAts = range(10000)
times = range(1000)
corrects = [0, 1]
generatedData = []

for tag in tags:
    for problemIdSet in problemIdSets:
        userIdSet = userIdSets[userIdSetIndex % 5]
        userIdSetIndex += 1
        for userId in userIdSet:
            for problemId in problemIdSet:
                endAt = int(random.uniform(0, 10001))
                time = int(random.uniform(0, 1001))
                correct = 0
                rand = random.uniform(0, 1)
                if rand < 0.8:
                    correct = 1
                generatedData.append([tag, userId, problemId + tag * 100, endAt, time, correct])

savetxt("data/data.txt", array(generatedData), fmt="%d")
