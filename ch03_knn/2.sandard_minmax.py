# 作者：liuyankai
# 时间：2025年08月03日20时50分28秒
# liuyankai23@mails.ucas.ac.cn
from sklearn.preprocessing import MinMaxScaler,StandardScaler
X = [[2, 1], [3, 1], [1, 4], [2, 6]]
X = MinMaxScaler(feature_range=(-1,1)).fit_transform(X)
print(X)

X = [[2, 1], [3, 1], [1, 4], [2, 6]]
X = StandardScaler().fit_transform(X)
print(X)