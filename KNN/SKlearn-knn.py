from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#设置变量
x_features = np.array([
    [1,0,1,0,1],
    [3,4,3,4,3],
    [0,1,0,1,0],
    [4,3,4,3,4]
])

y_labels = np.array([0,1,0,1])

#配置模型
model = KNeighborsClassifier(n_neighbors=1)

model.fit(x_features, y_labels)

#模型预测
X = np.array([[1,1,0,0,1]])

for i in range(10):
    X= np.random.randint(0,2,size=5)
    X=X.reshape((1,5))
    print(model.predict_proba(X))


#knn在学习什么
#维度爆炸
#如何学习非线性分类

