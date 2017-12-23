# -*- coding: UTF-8 -*-
# filename: ProblemRecord date: 2017/12/22 20:55  
# author: FD


class ProblemRecord(object):
    userId = 0
    problemId = 0
    time = 0
    endAt = 0
    correct = 0

    def __init__(self, userId, problemId, time, endAt, correct):
        self.userId = userId
        self.problemId = problemId
        self.time = time
        self.endAt = endAt
        self.correct = correct

    pass
