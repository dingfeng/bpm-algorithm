# -*- coding: UTF-8 -*-
# filename: algorithom date: 2017/12/22 21:05  
# author: FD
from numpy import *
import matplotlib as plt
# conf preprocess
MaxTimeMargin = 1000
userIdCount = 5
ProblemCount = 10000
MinTotalCount = 5
ProblemJoinProblem=[]

# 排序函数 升序排序
def dataComp(x, y):
    return x[4] > y[4]


def getAllData():
    userIds = range(userIdCount)
    problemIds = range(ProblemCount)
    endAts = range(1000000)
    times = range(10000)
    corrects = range(2)
    allData = []
    for j in range(userIds.count()):
        allData.append([])
    dataCount = 10000
    for i in xrange(dataCount):
        userId = int(np.random.uniform(0, userIds.count()))  # 随机取得一个用户编号
        problemId = int(np.random.uniform(0, problemIds.count()))  # 随机选取问题编号
        endAt = int(np.random.uniform(0, endAts.count()))  # 随机选取答题完成时间
        time = int(np.random.uniform(0, times.count()))  # 随机选取答题所需时间
        correct = int(np.random.uniform(corrects.count()))  # 随机选取答题是否正确
        userData = allData[userId]
        userData.append([problemId, endAt, time, correct])
    return allData


# 预处理得到一个同时错误和同时正确的关联概率 关联概率将作为聚类的distance
def preProcess(allData):
    global MaxTimeMargin
    global MinTotalCount
    global ProblemJoinProblem
    # map
    ProblemJoinProblem = [[0] * ProblemCount] * ProblemCount
    totalProblemJoinProblem = [[0] * ProblemCount] * ProblemCount
    # 每个用户的答题记录按时间排序
    for dataIndex, data in enumerate(allData):
        # 统计总共同时出现次数
        dataSize = data.count()
        for firstDataIndex in range(dataSize):
            for secondDataIndex in range(firstDataIndex + 1, dataSize):
                xProblem = data[firstDataIndex]
                yProblem = data[secondDataIndex]
                # 判断时间间隔是否超过阈值
                if abs(xProblem[2] - yProblem[2]) > MaxTimeMargin:
                    totalProblemJoinProblem[xProblem[1], yProblem[1]] += 1
                    totalProblemJoinProblem[yProblem[1], xProblem[1]] += 1

    for i in xrange(ProblemCount):
        for j in xrange(ProblemCount):
            if (totalProblemJoinProblem[i][j] < MinTotalCount):
                totalProblemJoinProblem[i][j] = 0

        # 统计同时正确和同时错误关联次数
        ProblemIds = [[] * 2]
        for oneDataRecord in data:
            correct = oneDataRecord[4]
            ProblemIds[correct].append(oneDataRecord)
        # 关联题目
        for oneProblemIdArray in ProblemIds:
            oneProblemIdArraySize = oneProblemIdArray.count()
            for xProblemIndex in range(oneProblemIdArraySize):
                for yProblemIndex in range(xProblemIndex + 1, oneProblemIdArraySize):
                    xProblem = oneProblemIdArray[xProblemIndex]
                    yProblem = oneProblemIdArray[yProblemIndex]
                    # 判断时间是否超过某一个阈值 如果超过进行处理
                    if abs(xProblem[2] - yProblem[2]) > MaxTimeMargin:
                        ProblemJoinProblem[xProblem[1], yProblem[1]] += 1
                        ProblemJoinProblem[yProblem[1], xProblem[1]] += 1
        # 计算同时正确和同时错误关联概率
        for i in range(ProblemCount):
            for j in range(ProblemCount):
                ProblemJoinProblem[i][j] = float(ProblemJoinProblem[i][j]) / totalProblemJoinProblem[i][j]
    return ProblemJoinProblem

#计算两个题目之间的距离
def dist(X,Y):
    return ProblemJoinProblem[X][Y]
#随机生成初始的质心()
def randCent(dataSet,k):
    n=shape(dataSet)[1]
    centroids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(array(dataSet))-minJ)
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)
    return centroids

#calculate distance
def kMeans(dataSet, k, distMeas=dist, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print centroids
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment

#显示结果
def show(dataSet, k, centroids, clusterAssment):
    from matplotlib import pyplot as plt
    numSamples, dim = dataSet.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in xrange(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)
    plt.show()


def main():
    # dataMat = mat(loadDataSet('testSet.txt'))
    # myCentroids, clustAssing = kMeans(dataMat, 4)
    # print myCentroids
    # show(dataMat, 4, myCentroids, clustAssing)
    pass


if __name__ == '__main__':
    main()