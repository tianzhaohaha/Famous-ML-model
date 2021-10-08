import numpy as np


class perception(object):
    def __init__(self, W_num):
        self.W_num=W_num
        self.Weight=np.random.random((W_num+1,1))


    def update(self,fea, lab):
        feature=np.array(fea)
        label=np.array(lab)
        output= self.predict(feature)
        if (label*output)<0:
            self.Weight=self.Weight+label.dot(feature.T).reshape((3,1))


    def predict(self,fea):
        feature = np.array(fea)
        #return self.activater(feature.dot(self.Weight)[0])
        return feature.dot(self.Weight)[0]

"""
    def activater(self,x):
        if x>0:
            return 1
        else:
            return -1
"""


if __name__ == '__main__':
    feature = np.random.randint(1, 4, size=(1,2))
    for i in range(19):
        if ((i%2)==0):
            feature1=np.random.randint(5, 8, size=(1,2))
            feature=np.vstack([feature,feature1])
        else:
            feature1 = np.random.randint(1, 4, size=(1, 2))
            feature = np.vstack([feature, feature1])

    labels=np.array([-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1])

    p=perception(2)

    feature3=np.ones((20,1))
    feature=np.concatenate([feature,feature3],axis=1)
    i=0
    for i in range(20):
        p.update(feature[i],labels[i])


    print(p.predict([7,8,1]))