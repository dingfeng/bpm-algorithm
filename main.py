# -*- coding: UTF-8 -*-
# filename: main date: 2017/12/24 19:23  
# author: FD
import numpy as np
from KMeansPlusPlus import KMeansPlusPlus

ProblemCount = 10;


# 根据做题记录生成数据
def main():
    data = np.array(np.loadtxt("data/data.txt", dtype=int))
    tags = np.unique(data[:, 0])
    tagDatas = []
    kMeansResults = []
    centCount = ProblemCount / np.shape(tags)[0]
    for tag in tags:
        tagData = data[np.where(data[:, 0] == tag), :][0][:, 1:]
        tagDatas.append(tagData)
        tagData[:, 2] = tagData[:, 2] - tag * 100  #problem Id 在0~100之间
        kmeans = KMeansPlusPlus(maxTimeMargin=0, problemCount=np.shape(tagData)[0], centCount=centCount)
        result = kmeans.process(tagData)
        kMeansResults.append(result)

    print "tags = ", tags
    pass


if __name__ == '__main__':
    main()
