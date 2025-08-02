import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
import joblib

heart_disease_data = pd.read_csv('../../data/heart_disease.csv')

# 数据清洗
heart_disease_data.dropna()

heart_disease_data.info()

print(heart_disease_data.head())
# 数据集划分
X = heart_disease_data.drop('是否患有心脏病', axis=1)
y = heart_disease_data['是否患有心脏病']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 特征工程
# 数值型特征
numerical_features = ['年龄', '静息血压', '胆固醇', '最大心率', '运动后的ST下降', '主血管数量']
# 类别性特征
categorical_features = ['胸痛类型', '静息心电图结果', '峰值ST段的斜率', '地中海贫血']
# 二元类别特征
binary_features = ['性别', '空腹血糖', '运动性心绞痛']

transformer = ColumnTransformer(
    # (名称,操作,特征列表)
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('bin', 'passthrough', binary_features)
    ]
)
# 执行特征转换
x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_test)

# 创建模型并训练
# knn = KNeighborsClassifier(n_neighbors=3)
knn = KNeighborsClassifier()
param_grid={'n_neighbors':list(range(1,11))}
knn = GridSearchCV(knn,param_grid=param_grid,cv=10)
knn.fit(x_train, y_train)

# # 测试
# # 准确率
# print(knn.score(x_test, y_test))

# # 保存模型对象到二进制文件
# joblib.dump(knn, 'knn_heart_disease.joblib')
#
# # 从文件中加载模型
# knn_load = joblib.load('knn_heart_disease.joblib')
# print(knn_load.score(x_test, y_test))
# print(knn.cv_results_)
# print(knn.best_estimator_)
# print(knn.best_params_)
knn = knn.best_estimator_
print(knn.score(x_test, y_test))