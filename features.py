from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


#导入IRIS数据集
iris = load_iris()

#特征矩阵
iris.data

#目标向量
iris.target

print("**")
print(iris.target)
print(iris.data)

print(StandardScaler().fit_transform(iris.data))

a=SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)
# print(a)


