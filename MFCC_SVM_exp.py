import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import time


start = time.time()

np.set_printoptions(threshold=np.inf)
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

data = pd.read_csv('mfcc_64_16.csv', header=None)
print(data.shape)

X = data.iloc[:, :42].values
print(X.shape)
y = data.iloc[:, 42].values
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y)
# model = SVC(kernel='rbf', C=100, gamma=0.01)
model = SVC(cache_size=2048)
model.fit(X_train, y_train)

fit_end = time.time()
print("Running time: %d minutes %d seconds " % (((fit_end-start)//60), ((fit_end-start) % 60)))

y_pre = model.predict(X_test)
# print('Accuracy of Classification:', model.score(X_test, y_test))
print("Accuracy of Classification:", accuracy_score(y_test, y_pre)*100, '%')
# print('confusion_matrix', confusion_matrix(y_test, y_pre))

cm = confusion_matrix(y_test, y_pre)
print(cm)
cm_dis = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_dis.plot()  # doctest: +SKIP
# print('precision_score', precision_score(y_test, y_pre))
# print('recall_score', recall_score(y_test, y_pre))
# print('f1_score', f1_score(y_test, y_pre))
print('classification_report', classification_report(y_test, y_pre))


plt.matshow(cm, cmap=plt.cm.gray)
plt.show()
row_sum = np.sum(cm, axis=1)
erro_matrix = cm/row_sum
np.fill_diagonal(erro_matrix, 0)  #将对角线的值填充为0
# print(erro_matrix)
plt.matshow(erro_matrix, cmap=plt.cm.gray)   #输出多元分类结果时所输出的错误结果
plt.show()

# print("准确率：", model.score(X_test, y_test)*100, '%')

end = time.time()
print("Running time: %d minutes %d seconds " % (((end-start)//60), ((end-start) % 60)))
