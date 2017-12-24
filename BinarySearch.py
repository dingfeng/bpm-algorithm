# -*- coding: UTF-8 -*-
# filename: BinarySearch date: 2017/12/24 22:54  
# author: FD 

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
        print "mid= ", mid
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


def main():
    list = [0.65000000000000002, 0.69999999999999996, 0.69999999999999996, 0.69999999999999996, 0.75, 0.75, 0.75, 0.75,
            0.80000000000000004, 0.80000000000000004, 0.80000000000000004, 0.80000000000000004, 0.84999999999999998,
            0.84999999999999998, 0.84999999999999998, 0.90000000000000002, 0.90000000000000002, 0.90000000000000002,
            0.94999999999999996, 1.0]
    index = binarySearch(list, 0.6662)
    print index
    pass


if __name__ == '__main__':
    main()
