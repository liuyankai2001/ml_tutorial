# 作者：liuyankai
# 时间：2025年08月03日17时44分29秒
# liuyankai23@mails.ucas.ac.cn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report

plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
import matplotlib
matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg'、'Agg' 视图不显示但可以保存


label = ["猫", "狗"] # 标签
y_true = ["猫", "猫", "猫", "猫", "猫", "猫", "狗", "狗", "狗", "狗"] # 真实值
y_pred1 = ["猫", "猫", "狗", "猫", "猫", "猫", "猫", "猫", "狗", "狗"] # 预测值
matrix1 = confusion_matrix(y_true, y_pred1, labels=label) # 混淆矩阵
print(pd.DataFrame(matrix1, columns=label, index=label))
sns.heatmap(matrix1, annot=True, fmt='d', cmap='Greens')

accuracy = accuracy_score(y_true, y_pred1)
print("准确率：",accuracy)
precision = precision_score(y_true, y_pred1, pos_label="猫")
print("精确率：",precision)
recall = recall_score(y_true, y_pred1, pos_label="猫")
print("召回率：",recall)
f1 = f1_score(y_true, y_pred1, pos_label="猫")
print("f1_score：",f1)
report = classification_report(y_true, y_pred1, labels=label, target_names=None)
print("report：",report)
