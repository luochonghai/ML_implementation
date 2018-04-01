import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import logistic_regression
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut

'''data visualization via seaborn'''
sns.set(style = "white",color_codes = True)
iris = pd.read_csv("D:\\FDU\\Template\\CS\\DataSet\\tic-tac-toe.csv")
iris["tl"] = iris.tl.map({"o":1,"b":2,"x":3})
iris["tm"] = iris.tm.map({"o":1,"b":2,"x":3})
iris["tr"] = iris.tr.map({"o":1,"b":2,"x":3})
iris["ml"] = iris.ml.map({"o":1,"b":2,"x":3})
iris["mm"] = iris.mm.map({"o":1,"b":2,"x":3})
iris["mr"] = iris.mr.map({"o":1,"b":2,"x":3})
iris["bl"] = iris.bl.map({"o":1,"b":2,"x":3})
iris["bm"] = iris.bm.map({"o":1,"b":2,"x":3})
iris["br"] = iris.br.map({"o":1,"b":2,"x":3})
iris["label"] = iris.label.map({"positive":1,"negative":0})
iris.to_csv("D:\\FDU\\Template\\CS\\DataSet\\tic-tac-toe_1.csv")
iris = pd.read_csv("D:\\FDU\\Template\\CS\\DataSet\\tic-tac-toe_1.csv")
X0 = iris.values[1:959,0:10]
y0 = iris.values[1:959,10]

dataset_temp = logistic_regression.randomize_data(X0,y0)
X = dataset_temp[0]
y = dataset_temp[1]

'''logistic regression using sklearn'''
log_model = LogisticRegression()
m = np.shape(X)[0]

#k-fold CV
k = 10
y_pred = cross_val_predict(log_model,X,y,cv = k)
print(metrics.accuracy_score(y,y_pred))

#LOO
loo = LeaveOneOut()
correct_judge = 0
for train,test in loo.split(X):
	log_model.fit(X[train],y[train])#fitting
	y_predict = log_model.predict(X[test])
	if y_predict == y[test]:
		correct_judge += 1
print(correct_judge/np.shape(X)[0])
