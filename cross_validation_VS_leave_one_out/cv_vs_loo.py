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
#iris = pd.read_csv("D:\\FDU\\Template\\CS\\DataSet\\Iris.csv")
#iris["species"] = iris.species.map({"Iris-versicolor":3,"Iris-setosa":1,"Iris-virginica":2})
#iris.to_csv("D:\\FDU\\Template\\CS\\DataSet\\Iris_1.csv")
iris = pd.read_csv("D:\\FDU\\Template\\CS\\DataSet\\Iris.csv")
#sns.swarmplot(x = "species",y = "sepal_length",data = iris)
#plt.show()
X0 = iris.values[1:151,0:4]
y0 = iris.values[1:151,4]
dataset_temp = logistic_regression.randomize_data(X0,y0)
X = dataset_temp[0]
y = dataset_temp[1]
#iris.plot(kind = "scatter",x = "sepal_length",y = "sepal_width")
sns.pairplot(iris,hue = 'species')
plt.show()

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
