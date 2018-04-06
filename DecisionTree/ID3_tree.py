# -*-coding: utf-8 -*

#using pandas dataframe for .csv read 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import decision_tree
from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname='C:\\Windows\\Fonts\\simhei.ttf',size = 10)
sns.set(font = myfont.get_name())

data_file_encode = 'gb18030'
with open("D:\\FDU\\Template\\CS\\Machine Learning\\周志华西瓜书编程练习\\ch4决策树\\4.3_ID3_tree\\watermelon_3a.csv",mode = 'r',encoding = data_file_encode) as data_file:
	df = pd.read_csv(data_file)

df = df.reindex(np.random.permutation(df.index))

sns.FacetGrid(df, hue="好瓜", size=5).map(plt.scatter, "密度", "含糖量").add_legend() 
plt.show()

f2 = plt.figure(2)
plt.subplot(221)
sns.swarmplot(x = "纹理", y = '密度', hue = "好瓜", data = df,size = 5)
plt.subplot(222)
sns.swarmplot(x = "敲声", y = '密度', hue = "好瓜", data = df,size = 5)
plt.subplot(223)
sns.swarmplot(x = "色泽", y = '含糖量', hue = "好瓜", data = df,size = 5)
plt.subplot(224)
sns.swarmplot(x = "敲声", y = '含糖量', hue = "好瓜", data = df,size = 5)
plt.show()
'''implementation of ID3'''
root = decision_tree.TreeGenerate(df)

accuracy_scores = []

#k-folds cross validation
n = len(df.index)
k = 5
for i in range(k):
	m = int(n/k)
	test = []
	for j in range(i*m,i*m+m):
		test.append(j)

	df_train = df.drop(test)
	df_test = df.iloc[test]
	root = decision_tree.TreeGenerate(df_train)

	#test the accuracy
	pred_true = 0
	for i in df_test.index:
		label = decision_tree.Predict(root,df[df.index == i])
		if label == df_test[df_test.columns[-1]][i]:
			pred_true += 1

	accuracy = pred_true/len(df_test.index)
	accuracy_scores.append(accuracy)

#print the prediction accuracy result
accuracy_sum = 0
print("accuracy: ",end = "")
for i in range(k):
	print("%.3f  "%accuracy_scores[i],end = "")
	accuracy_sum += accuracy_scores[i]
print("\naverage accuracy: %.3f"%(accuracy_sum/k))

#dicision tree visualization using pydotplus.graphviz
root = decision_tree.TreeGenerate(df)

decision_tree.DrawPNG(root,"decision_tree_ID3.png")

