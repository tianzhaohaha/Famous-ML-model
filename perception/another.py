# encoding=utf8
import numpy as np


# 构建感知机算法
class Perceptron(object):
    def __init__(self, learning_rate=0.01, max_iter=200):
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, data, label):
        '''
        input:data(ndarray):训练数据特征
              label(ndarray):训练数据标签
        output:w(ndarray):训练好的权重
               b(ndarry):训练好的偏置
        '''
        # 编写感知机训练方法，w为权重，b为偏置
        self.w = np.array([1.] * data.shape[1])
        self.b = np.array([1.])
        # ********* Begin *********#
        flag = True
        num = 0
        while (num < self.max_iter):
            flag = True
            for j in range(data.shape[0]):
                if (label[j] * (self.w.dot(np.array(data[j]).T) + self.b)[0] <= 0):
                    self.w = self.w + self.lr * (np.array(label[j]).dot(data[j].T))
                    self.b = self.b + self.lr * label[j]
                    flag = False;

            if flag:
                break
            num += 1
        return self.w, self.b

        # ********* End *********#

    def predict(self, data):
        '''
        input:data(ndarray):测试数据特征
        output:predict(ndarray):预测标签
        '''
        # ********* Begin *********#
        # i=0
        # for i in range(data.shape[0])
        a = []
        for i in data:
            if ((self.w.dot(i.T) + self.b)[0] >= 0):
                a.append(1)
            else:
                a.append(-1)

        predict = np.array(a)

        # ********* End *********#
        return predict
































