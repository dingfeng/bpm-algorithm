# -*- coding: UTF-8 -*-
# filename: KmeansPlusPlus date: 2017/12/24 18:57  
# author: FD 
import numpy as np



class KMeansPlusPlus(object):
    def __init__(self, maxTimeMargin, problemCount, centCount):
        self.MaxTimeMargin = maxTimeMargin
        self.ProblemCount = problemCount
        self.ProblemJoinProblem = np.array([[0] * problemCount] * problemCount)
        self.totalProblemJoinProblem = np.array([[0] * problemCount] * problemCount)
        self.centCount = centCount

    # 排序函数 升序排序


    # 预处理得到一个同时错误和同时正确的关联概率 关联概率将作为聚类的distance
    def preProcess(self, allData):
        userIds = np.unique(allData[:, 0])
        for userId in userIds:
            userIdRecordIndexes = np.where(allData[:, 0] == userId)
            userIdDatas = allData[userIdRecordIndexes, :][0]
            self.preProcessOneUserData(userIdDatas)
        for i in range(self.ProblemCount):
            for j in range(self.ProblemCount):
                if (self.ProblemJoinProblem[i][j] != 0):
                    self.ProblemJoinProblem[i][j] = self.totalProblemJoinProblem[i][j] / float(
                        self.ProblemJoinProblem[i][j])
                else:
                    self.ProblemJoinProblem[i][j] = 10000
            self.ProblemJoinProblem[i][i] = 0

    # 处理一个用户的数据i
    def preProcessOneUserData(self, oneUserData):
        # 每个用户的答题记录按时间排序
        # 统计总共同时出现次数

        dataSize = np.shape(oneUserData)[0]
        for firstDataIndex in range(dataSize):
            for secondDataIndex in range(firstDataIndex + 1, dataSize):
                xProblem = oneUserData[firstDataIndex]
                yProblem = oneUserData[secondDataIndex]
                # 判断时间间隔是否超过阈值
                if abs(xProblem[2] - yProblem[2]) > self.MaxTimeMargin:
                    xProblemId = int(xProblem[1])
                    yProblemId = int(yProblem[1])
                    self.totalProblemJoinProblem[xProblemId, yProblemId] += 1
                    self.totalProblemJoinProblem[yProblemId, xProblemId] += 1
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
                    if abs(xProblem[2] - yProblem[2]) > self.MaxTimeMargin:
                        self.ProblemJoinProblem[xProblem[1], yProblem[1]] += 1
                        self.ProblemJoinProblem[yProblem[1], xProblem[1]] += 1
        # 计算同时正确和同时错误关联概率

        return self.ProblemJoinProblem

    # 计算两个题目之间的距离
    def dist(self, X, Y):
        return self.ProblemJoinProblem[int(X[0][0])][int(Y[0][0])]

    # 随机生成初始的质心()
    def kMeansPlusPlusCreateCent(self, dataSet, k):
        # 离散取点
        centroids = []
        maxProblemId = np.max(dataSet[:, 0])
        preCent = int(np.random.uniform(0, maxProblemId + 1))
        centroids.append([preCent])
        for i in range(1, k):
            # 每个点计算 D x
            sumDx = 0
            problemDists = []
            for problemId in range(self.ProblemCount):
                # 取最近的中心点
                minDist = 100000
                for j in range(i):
                    cent = centroids[j][0]
                    centDist = self.dist([[cent]], [[problemId]])
                    if centDist < minDist:
                        minDist = centDist
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
    def kMeansPlusPlus(self, dataSet, k, distMeas=dist, createCent=kMeansPlusPlusCreateCent):
        m = np.shape(dataSet)[0]
        clusterAssment = np.mat(np.zeros((m, 2)))  # create mat to assign data points
        # to a centroid, also holds SE of each point
        centroids = np.mat(createCent(self,dataSet, k))
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):  # for each data point assign it to the closest centroid
                minDist = 10001
                minIndex = -1
                for j in range(k):
                    distJI = distMeas(self,centroids[j, :], dataSet[i, :])
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
                        distSum += distMeas(self,[pt], [pt1])
                        if (distSum > minDistSum):
                            break
                    if (distSum < minDistSum):
                        minDistSum = distSum
                        minPt = pt
                    else:
                        continue
                centroids[cent, :] = np.array([minPt])
        return centroids, clusterAssment

    # 处理数据
    def process(self, data):
        self.preProcess(data)
        dataSet = np.mat([[i] for i in range(self.ProblemCount)])
        myCentroids, clustAssing = self.kMeansPlusPlus(dataSet, self.centCount)
        return myCentroids, clustAssing
