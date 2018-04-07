# -*- coding:utf-8 -*

import pandas as pd
import decision_tree
import random
data_file_encode = 'gb18030'

with open("D:\\FDU\\Template\\CS\\Machine Learning\\周志华西瓜书编程练习\\ch4决策树\\4.4_CART_tree\\watermelon_3a.csv",mode = 'r',encoding = data_file_encode) as data_file:
	df = pd.read_csv(data_file)

data_list = [i for i in range(0,17)]
index_train = random.sample(data_list,10)

df_test = df.iloc[index_train]
df_train = df.drop(index_train)

#generate a full tree
root = decision_tree.TreeGenerate(df_train)
decision_tree.DrawPNG(root,"decision_tree_full.png")
print("accuracy of full tree:%.3f"%decision_tree.PredictAccuracy(root,df_test))

#pre-prune
root = decision_tree.Preprune(df_train,df_test)
decision_tree.DrawPNG(root,"decision_tree_pre.png")
print("accuracy of pre-prune tree:%.3f"%decision_tree.PredictAccuracy(root,df_test))

#post-prune
root = decision_tree.TreeGenerate(df_train)
decision_tree.PostPrune(root,df_test)
decision_tree.DrawPNG(root,"decision_tree_post.png")
print("accuracy of post-prune tree:%.3f"%decision_tree.PredictAccuracy(root,df_test))

#k-folds cross validation
accuracy_scores = []
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
	decision_tree.PostPrune(root,df_test)

	pred_true = 0
	for i in df_test.index:
		label = decision_tree.Predict(root,df[df.index == i])
		if label == df_test[df_test.columns[-1]][i]:
			pred_true += 1

	accuracy = pred_true/len(df_test.index)
	accuracy_scores.append(accuracy)

accuracy_sum = 0
print("accuracy: ",end = "")
for i in range(k):
	print("%.3f"%accuracy_scores[i],end = "")
	accuracy_sum += accuracy_scores[i]
print("\naverage accuracy:%.3f"%(accuracy_sum/k))
