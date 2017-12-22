# -*- coding: UTF-8 -*-
# filename: ProblemRecord date: 2017/12/22 20:55  
# author: FD


class ProblemRecord(object):
    userId = 0
    problemId = 0
    time = ""
    correct = False

    def __init__(self, userId, problemId, time, correct):
        self.userId = userId
        self.problemId = problemId
        self.time = time
        self.correct = correct

    pass
