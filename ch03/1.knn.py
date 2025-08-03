# 作者：liuyankai
# 时间：2025年08月03日20时46分57秒
# liuyankai23@mails.ucas.ac.cn
from sklearn.neighbors import KNeighborsRegressor,KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=2)
X = [[2, 1], [3, 1], [1, 4], [2, 6]] # 特征
y = [0, 0, 1, 1]
knn.fit(X,y)
print(knn.predict([[4, 9]]))

knn1 = KNeighborsRegressor(n_neighbors=2)
X1 = [[2, 1], [3, 1], [1, 4], [2, 6]]
y1 = [0.5, 0.33, 4, 3]
knn.fit(X1, y1)
print(knn.predict([[4, 9]]))
