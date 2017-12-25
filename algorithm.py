# -*- coding: UTF-8 -*-
# filename: algorithom date: 2017/12/22 21:05  
# author: FD

import matplotlib as plt
import numpy as np
from scipy import stats

# conf preprocess
MaxTimeMargin = 0
ProblemCount = 100
ProblemJoinProblem = np.array([[0] * ProblemCount] * ProblemCount)
totalProblemJoinProblem = np.array([[0] * ProblemCount] * ProblemCount)
# 试卷难度
Difficuty = 0.6


# 初始化变量
def init(maxTimeMargin, problemCount):
    global MaxTimeMargin
    global totalProblemJoinProblem
    global ProblemJoinProblem
    global ProblemCount
    MaxTimeMargin = maxTimeMargin
    ProblemCount = problemCount
    ProblemJoinProblem = np.array([[0] * ProblemCount] * ProblemCount)
    totalProblemJoinProblem = np.array([[0] * ProblemCount] * ProblemCount)


# 排序函数 升序排序
def dataComp(x, y):
    return x[4] > y[4]


# 预处理得到一个同时错误和同时正确的关联概率 关联概率将作为聚类的distance
def preProcess(allData):
    userIds = np.unique(allData[:, 0])
    for userId in userIds:
        userIdRecordIndexes = np.where(allData[:, 0] == userId)
        userIdDatas = allData[userIdRecordIndexes, :][0]
        preProcessOneUserData(userIdDatas)
    for i in range(ProblemCount):
        for j in range(ProblemCount):
            if (ProblemJoinProblem[i][j] != 0):
                ProblemJoinProblem[i][j] = totalProblemJoinProblem[i][j] / float(ProblemJoinProblem[i][j])
            else:
                ProblemJoinProblem[i][j] = 10000
        ProblemJoinProblem[i][i] = 0


# 处理一个用户的数据i
def preProcessOneUserData(oneUserData):
    global ProblemJoinProblem
    global totalProblemJoinProblem
    # 每个用户的答题记录按时间排序
    # 统计总共同时出现次数

    dataSize = np.shape(oneUserData)[0]
    for firstDataIndex in range(dataSize):
        for secondDataIndex in range(firstDataIndex + 1, dataSize):
            xProblem = oneUserData[firstDataIndex]
            yProblem = oneUserData[secondDataIndex]
            # 判断时间间隔是否超过阈值
            if abs(xProblem[2] - yProblem[2]) > MaxTimeMargin:
                xProblemId = int(xProblem[1])
                yProblemId = int(yProblem[1])
                totalProblemJoinProblem[xProblemId, yProblemId] += 1
                totalProblemJoinProblem[yProblemId, xProblemId] += 1
    # 统计同时正确和同时错误关联次数
    ProblemIds = [[], []]
    for oneDataRecord in oneUserData:
        correct = oneDataRecord[4]
        ProblemIds[correct].append(oneDataRecord)
    # 关联题目
    for oneProblemIdArray in ProblemIds:
        oneProblemIdArraySize = np.shape(oneProblemIdArray)[0]
        for xProblemIndex in range(oneProblemIdArraySize):
            for yProblemIndex in range(xProblemIndex + 1, oneProblemIdArraySize):
                xProblem = oneProblemIdArray[xProblemIndex]
                yProblem = oneProblemIdArray[yProblemIndex]
                # 判断时间是否超过某一个阈值 如果超过进行处理
                if abs(xProblem[2] - yProblem[2]) > MaxTimeMargin:
                    ProblemJoinProblem[xProblem[1], yProblem[1]] += 1
                    ProblemJoinProblem[yProblem[1], xProblem[1]] += 1
    # 计算同时正确和同时错误关联概率

    return ProblemJoinProblem


# 计算两个题目之间的距离
def dist(X, Y):
    return ProblemJoinProblem[int(X[0][0])][int(Y[0][0])]


# 随机生成初始的质心()
def kMeansPlusPlusCreateCent(dataSet, k):
    # 离散取点
    centroids = []
    maxProblemId = np.max(dataSet[:, 0])
    preCent = int(np.random.uniform(0, maxProblemId + 1))
    centroids.append([preCent])
    for i in range(1, k):
        # 每个点计算 D x
        sumDx = 0
        problemDists = []
        for problemId in range(ProblemCount):
            # 取最近的中心点
            minDist = 100000
            minCent = -1
            for j in range(i):
                cent = centroids[j][0]
                centDist = dist([[cent]], [[problemId]])
                if centDist < minDist:
                    minDist = centDist
                    minCent = cent
            problemDists.append(minDist)
            sumDx += minDist
        randProblemDist = sumDx * np.random.uniform(0, 1)
        for index, value in enumerate(problemDists):
            randProblemDist -= value
            if randProblemDist <= 0:
                centroids.append([index])
                break

    return centroids


# calculate distance
def kMeansPlusPlus(dataSet, k, distMeas=dist, createCent=kMeansPlusPlusCreateCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = np.mat(createCent(dataSet, k))
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = 10001
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist

        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            # print "ptsInClust= ", cent, " []=", ptsInClust
            # 修改每个类的中心点 提高内聚性 内部距离和最小
            minDistSum = float('Inf')
            minPt = -1
            for pt in ptsInClust:
                distSum = 0
                for pt1 in ptsInClust:
                    distSum += dist([pt], [pt1])
                    if (distSum > minDistSum):
                        break
                if (distSum < minDistSum):
                    minDistSum = distSum
                    minPt = pt
                else:
                    continue
            centroids[cent, :] = np.array([minPt])
    return centroids, clusterAssment


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
    print "list=", list
    return list


# 二分法区间搜索
def binarySearch(list, num):
    start = 0
    end = list.__len__() - 1
    if num <= list[0]:
        return 0
    elif num >= list[end]:
        return end
    while end >= start:
        mid = (end + start) / 2
        midValue = list[mid]
        # 先判断是否已经找到
        if num > midValue and num < list[mid + 1]:
            if abs(midValue - num) > abs(list[mid + 1] - num):
                return mid + 1
            else:
                return mid
        elif num > midValue:
            start = mid + 1
        elif num < midValue:
            end = mid - 1


# 比较函数
def reverseComp(x, y):
    if x[1] > y[1]:
        return 1
    elif x[1] < y[1]:
        return -1
    return 0


def main():
    ExamProblemNum = 10
    data = np.array(np.loadtxt("data/data.txt", dtype=int))
    problemNum = np.unique(data[:, 2]).__len__()
    tags = np.unique(data[:, 0])
    tagDatas = []
    kMeansResults = []
    centCount = ExamProblemNum / np.shape(tags)[0]
    problemSets = []
    for tag in tags:
        init(0, 100)
        tagData = data[np.where(data[:, 0] == tag), :][0][:, 1:]
        tagDatas.append(tagData)
        tagData[:, 1] = tagData[:, 1] - tag * 100  # problem Id 在0~100之间
        preProcess(tagData)
        dataSet = np.mat([[i] for i in range(100)])
        myCentroids, clustAssing = kMeansPlusPlus(dataSet, centCount)
        kMeansResults.append([myCentroids, clustAssing])
        for i in range(centCount):
            clusterIndexes = np.where(clustAssing[:, 0] == i)[0] + tag * 100
            problemSets.append(clusterIndexes)
    # 计算data的争取率
    totalCounts = [0 for i in range(problemNum)]
    correctCounts = [0 for i in range(problemNum)]
    for line in data:
        problemId = line[2]
        correct = line[5]
        totalCounts[problemId] += 1
        if correct == 1:
            correctCounts[problemId] += 1
    correctRates = [0 for i in range(problemNum)]
    for i in range(problemNum):
        if totalCounts[problemId] > 0:
            correctRates[i] = float(correctCounts[i]) / totalCounts[i]
    correctSets = []
    for problemSet in problemSets:
        problemIndexCorrectSet = []
        for problemId in problemSet:
            correctRate = correctRates[problemId]
            problemIndexCorrectSet.append([problemId, correctRate])
        problemIndexCorrectSet.sort(reverseComp)
        correctSets.append(problemIndexCorrectSet)
    difficulties = generateDiffculties(ExamProblemNum)
    resultProblems = []
    for i in range(ExamProblemNum):
        difficulty = difficulties[i]
        correctSet = correctSets[i]
        searchedIndex = binarySearch(list(np.array(correctSet)[:, 1]), difficulty)
        resultProblems.append(correctSet[searchedIndex])
    print "resultProblems= ", resultProblems


if __name__ == '__main__':
    main()
