# encoding=utf8
import numpy as np


class kNNClassifier(object):
    def __init__(self, k):
        '''
        初始化函数
        :param k:kNN算法中的k
        '''
        self.k = k
        # 用来存放训练数据，类型为ndarray
        self.train_feature = None
        # 用来存放训练标签，类型为ndarray
        self.train_label = None

    def fit(self, feature, label):
        '''
        kNN算法的训练过程
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: 无返回
        '''

        # ********* Begin *********#
        self.train_feature = np.array(feature)
        self.train_label = np.array(label)

        # ********* End *********#

    def predict(self, feature):
        '''
        kNN算法的预测过程
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray或list
        '''

        # ********* Begin *********#
        j=0
        resultlist=[]
        for j in range(feature.shape[0]):
            distance = np.sqrt(np.square(self.train_feature - feature[j]).sum(1))
            sort = np.argsort(distance)
            sortk = sort[0:self.k]
            dissort = {}
            ksort = {}

            maxk = 0

            i = 0
            for i in range(self.k):
                if self.train_label[sortk[i]] in dissort:
                    dissort[self.train_label[sortk[i]]] = dissort[self.train_label[sortk[i]]] + distance[sortk[i]]
                else:
                    dissort[self.train_label[sortk[i]]] = distance[sortk[i]]
            i = 0
            for i in range(self.k):
                if self.train_label[sortk[i]] in ksort:
                    ksort[self.train_label[sortk[i]]] = ksort[self.train_label[sortk[i]]] + 1
                    if ksort[self.train_label[sortk[i]]] > maxk:
                        maxk = ksort[self.train_label[sortk[i]]]
                        result = self.train_label[sortk[i]]
                    elif ksort[self.train_label[sortk[i]]] == maxk:
                        if dissort[result] > dissort[self.train_label[sortk[i]]]:
                            result=self.train_label[sortk[i]]
                else:
                    ksort[self.train_label[sortk[i]]] = 1

            resultlist.append(result)
        return resultlist

        # ********* End *********#
