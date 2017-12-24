# -*- coding: UTF-8 -*-
# filename: TestAddProblem date: 2017/12/24 20:39  
# author: FD 
import numpy as np
from scipy.stats import stats


def generateDiffculties(size):
    count = 0
    list = []
    while count < size:
        oneNormValue = np.random.normal(0.5, 1, 1)
        while oneNormValue > 1 or oneNormValue < 0:
            oneNormValue = np.random.normal(0.5, 1, 1)
        list.append(oneNormValue[0])
        count += 1
    print "list size=", list.__len__()
    for i in range(list.__len__()):
        list[i] += 0.1
    list.sort()
    return list


