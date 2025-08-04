from sklearn.linear_model import LinearRegression

# 自变量，每周学习时长
X = [[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]]
# 因变量，数学考试成绩
y = [55, 65, 70, 75, 85, 50, 60, 72, 80, 58]

model = LinearRegression()

model.fit(X,y)

print(model.coef_)
print(model.intercept_)
print(model.predict([[11]]))