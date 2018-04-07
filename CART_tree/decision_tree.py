# -*- coding:utf-8 -*
'''defination of decision node class
attr: attribution as parent node for new branchs
sub_attr: dict:{key,value}
	key:
		categoric:categoric attr_value
		continuous:  <= div_value for small part
					 > div_value for big part
	value:
		children(node class)
label: class label(the majority of current sample labels)'''

class Node(object):
	def __init__(self,attr_init = None,label_init = None,sub_attr_init = {}):
		self.attr = attr_init
		self.label = label_init
		self.sub_attr = sub_attr_init

'''calculate the information entropy of an attribution
@ param label_arr: ndarray,class label array of data_arr
@ return ent: the information entropy of current attribution
'''
def InfoEnt(label_arr):
	try:
		from math import log2
	except ImportError:
		print("module math.log2 not found")

	ent = 0
	n = len(label_arr)
	label_count = NodeLabel(label_arr)

	for key in label_count:
		#p75：information entropy formula(4.1)
		ent -= (label_count[key]/n)*log2(label_count[key]/n)
	return ent

'''calculate the information gain of attribution 
@param df: dataframe, the pandas dataframe of the data_set
@param index: the target attribution in df
@return info_gain: the information gain of current attribution
@return div_value: for discrete variable, value = 0;for continuous variable, value = t(the division value)'''
def InfoGain(df,index):
	info_gain = InfoEnt(df.values[:,-1])#info_gain for the whole label
	div_value = 0#div_value for continuous attribute

	n = len(df[index])#the number of sample
	#step1: for continuous variable using method of bisection
	if isinstance(df[index].iloc[0],float):
		sub_info_ent = {}#store the div_value(div) and it's subset entropy
		df = df.sort_values([index],ascending = 1)#sort via column
		df = df.reset_index(drop = True)

		data_arr = df[index]
		label_arr = df[df.columns[-1]]

		for i in range(n-1):
			#p84: formula (4.7)
			div = (data_arr[i]+data_arr[i+1])/2
			#p84: formual (4.8)
			sub_info_ent[div] = ((i+1)*InfoEnt(label_arr[0:i+1])/n)\
			+((n-i-1)*InfoEnt(label_arr[i+1:-1])/n)
		#our goal is to get the min subset entropy sum and its divide value
		div_value,sub_info_ent_max = min(sub_info_ent.items(),key = lambda x:x[1])
		info_gain -= sub_info_ent_max

	#step2: for discrete variable(categoric variable)
	else:
		data_arr = df[index]#(feature value)
		label_arr = df[df.columns[-1]]#(label)
		value_count = ValueCount(data_arr)

		for key in value_count:
			key_label_arr = label_arr[data_arr == key]
			#formula: discrete information gain (4.2)
			info_gain -= value_count[key]*InfoEnt(key_label_arr)/n

	return info_gain,div_value

'''find the optimal attributes of current data_set
@param df: the pandas dataframe of the data_set
@return opt_attr: the optimal attribution for branch
@return div_value: for discrete variable value = 0;for continuous variable value = t for bisection divide value'''
def OptAttr_Ent(df):
	info_gain = 0
	opt_attr = ""
	div_value = 0
	for attr_id in df.columns[1:-1]:
		info_gain_tmp,div_value_tmp = InfoGain(df,attr_id)
		if info_gain_tmp > info_gain:
			info_gain = info_gain_tmp
			opt_attr = attr_id
			div_value = div_value_tmp
	return opt_attr,div_value

'''calculate the appeared value for categoric attribute and its counts
@param data_arr: data array for an attribute
@return value_count: dict,the appeared value and 
its counts(how many types of value has appeared)'''
def ValueCount(data_arr):
	value_count = {}# store the number of value

	for label in data_arr:
		if label in value_count: 
			value_count[label] += 1
		else:
			value_count[label] = 1
	return value_count

'''calculate the appeared label and its number
@param label_arr: data array for class labels
@return label_count:dict, the appeared label and its number'''
def NodeLabel(label_arr):
	label_count = {}#store number of label
	for label in label_arr:
		if label in label_count:
			label_count[label] += 1
		else:
			label_count[label] = 1
	return label_count

'''create a new branch via recursion
@param df: the pandas dataframe of the data_set
@return root: Node, the root node of decision tree
'''
def TreeGenerate(df):
	#generate a new root code
	new_node = Node(None,None,{})
	label_arr = df[df.columns[-1]]

	#to check whether the attribution set is empty by checking if there are feature-duplicated samples
	attr_set_num = 0
	if df.shape[0] >= 2:
		df_dupli = df.iloc[:,1:-1]
		df_drop = df_dupli.drop_duplicates()
		attr_set_num = df_drop.shape[0]
	else:
		attr_set_num = len(label_arr)+1

	label_count = NodeLabel(label_arr)
	if label_count:#assume that the label_count is nonempty
		new_node.label = max(label_count,key = label_count.get)

		#end if there's only 1 class in current node data
		#end if attribution set is empty
		
		'''if len(label_count) == 1 or len(label_arr) == 0:'''
		if len(label_count) == 1 or attr_set_num == 1:
			return new_node

		#get the optimal attribution for a new branching
		new_node.attr,div_value = OptAttr_Gini(df)

		#recursion
		if div_value == 0:#categoric variable
			value_count = ValueCount(df[new_node.attr])
			for value in value_count:
				df_v = df[df[new_node.attr].isin([value])]#get sub set
				#delete current attribution
				df_v = df_v.drop(new_node.attr,1)
				new_node.sub_attr[value] = TreeGenerate(df_v)
		else:
			value_less = "<=%.3f"%div_value
			value_more = ">%.3f"%div_value
			df_v_less = df[df[new_node.attr] <= div_value]#get sub set
			df_v_more = df[df[new_node.attr] > div_value]

			new_node.sub_attr[value_less] = TreeGenerate(df_v_less)
			new_node.sub_attr[value_more] = TreeGenerate(df_v_more)
	return new_node

'''make prediciton based on root node
@param root: root node of the decision tree
@param df_sample: dataframe, a sample vector'''
def Predict(root,df_sample):
	try:
		import re#using regular expression to get the number in string
	except ImportError:
		print("module re not found!")

	while root.attr != None:
		#continuous variable
		if isinstance(df_sample[root.attr].iloc[0],float):
			#get the div_value from root.sub_attr
			for key in list(root.sub_attr):
				num = re.findall(r"\d+\.?\d*",key)
				div_value = float(num[0])
				break
			if df_sample[root.attr].values[0] <= div_value:
				key = "<=%.3f"%div_value
				root = root.sub_attr[key]
			else:
				key = ">%.3f"%div_value
				root = root.sub_attr[key]

		#categoric variable
		else:
			key = df_sample[root.attr].values[0]
			#check whether the attr_value is in the child branch
			if key in root.sub_attr:
				root = root.sub_attr[key]
			else:
				break
	return root.label

def TreeToGraph(i,g,root):
	'''build a graph from root node via
	@param i: node number in this tree
	@param g: pydotplus.graphviz.Dot() object
	@param root: the root node
	@return i: node number after modified
	@return g: pydotplus.graphviz.Dot() object after modified
	@return g_node: the current root node in graphviz
	'''
	try:
		from pydotplus import graphviz
	except ImportError:
		print("module pydotplus.graphviz not found")

	if root.attr == None:
		g_node_label = "Node:%d\n好瓜:%s"%(i,root.label)
	else:
		g_node_label = "Node:%d\n好瓜:%s\nattribute:%s"%(i,root.label,root.attr)
	g_node = i
	g.add_node(graphviz.Node(g_node,label = g_node_label,fontname = "Microsoft YaHei",fontsize = 10))

	for value in list(root.sub_attr):
		i,g_child = TreeToGraph(i+1,g,root.sub_attr[value])
		g.add_edge(graphviz.Edge(g_node,g_child,label = value,fontname = "Microsoft YaHei",fontsize = 10))
	return i,g_node


def DrawPNG(root,out_file):
	'''visualization of decision tree from root
	@param root: the root node
	@param out_file: str,name and path of output file'''
	try:
		from pydotplus import graphviz
	except ImportError:
		print("module pydotplus.graphviz not found")

	g = graphviz.Dot()#generation of new dot
	TreeToGraph(0,g,root)
	g2 = graphviz.graph_from_dot_data(g.to_string())
	g2.write_png(out_file)

'''calculate the gini value of an attribution
@param label_arr: ndarray,class label array of data_arr
@return gini: the information entropy of current attribution p79 (4.5)
'''
def Gini(label_arr):
	gini = 1
	n = len(label_arr)
	label_count = NodeLabel(label_arr)
	for key in label_count:
		gini -= (label_count[key]/n)*(label_count[key]/n)

	return gini

'''calculate the gini_index of attribution
@param df: dataframe,the pandas dataframe of the dataset
@param attr_id: the target attribution in df
@return gini_index: the gini index of the current attribution
@return div_value: for discrete variable, value = 0;for continuous variable,value = t(division value) 
'''
def GiniIndex(df,attr_id):
	gini_index = 0
	div_value = 0
	n = len(df[attr_id])

	#step1: for continuous variable using bisection
	if isinstance(df[attr_id].iloc[0],float):
		sub_gini = {}#to store the div_value and its subset gini_value
		df = df.sort_values([attr_id],ascending = 1)#sorted in column
		df = df.reset_index(drop = True)

		data_arr = df[attr_id]
		label_arr = df[df.columns[-1]]

		for i in range(n-1):
			div = (data_arr[i]+data_arr[i+1])/2
			sub_gini[div] = ( (i+1) * Gini(label_arr[0:i+1]) / n ) \
			+ ( (n-i-1) * Gini(label_arr[i+1:-1]) / n )
		div_value, gini_index = min(sub_gini.items(), key=lambda x: x[1])

		#step2:for discrete variable(categoric variable)
	else:
		data_arr = df[attr_id]
		label_arr = df[df.columns[-1]]
		value_count = ValueCount(data_arr)

		for key in value_count:
			key_label_arr = label_arr[data_arr == key]
			gini_index += value_count[key] * Gini(key_label_arr) / n

	return gini_index,div_value

'''optimal attribution selection in CART algorithm based on gini_index
@param df: the pandas dataframe of dataset
@return opt_attr: the optimal attribution for branch
@return div_value: for discrete variable,value = 0; for continuous variable,variable value = t for fisection
'''
def OptAttr_Gini(df):
	gini_index = float('Inf')
	opt_attr = ""
	div_value = 0
	for attr_id in df.columns[1:-1]:
		gini_index_tmp,div_value_tmp = InfoGain(df,attr_id)
		if gini_index_tmp < gini_index:
			gini_index = gini_index_tmp
			opt_attr = attr_id
			div_value = div_value_tmp

	return opt_attr,div_value

'''calculate accuracy of prediction on the test set
@param root: root node
@param df_test: dataframe, test data set
@return accuracy(float format)
'''
def PredictAccuracy(root,df_test):
	if len(df_test.index) == 0:
		return 0
	pred_true = 0
	for i in df_test.index:
		label = Predict(root,df_test[df_test.index == i])
		if label == df_test[df_test.columns[-1]][i]:
			pred_true += 1
	return pred_true/len(df_test.index)

'''pre-purning to generate a decision tree
@param df_train: dataframe, the training set
@param df_test: dataframe, the testing set to prune tree
@return root: root node of the tree
'''
def Preprune(df_train,df_test):
	new_node = Node(None,None,{})
	label_arr = df_train[df_train.columns[-1]]
	attr_set_num = 0
	if df_train.shape[0] >= 2:
		df_dupli = df_train.iloc[:,1:-1]
		df_drop = df_dupli.drop_duplicates()
		attr_set_num = df_drop.shape[0]
	else:
		attr_set_num = len(label_arr)+1

	label_count = NodeLabel(label_arr)
	if label_count:
		new_node.label = max(label_count,key = label_count.get)

		#end if there's only 1 class in current node data
		#end if attribution set is empty
		
		'''if len(label_count) == 1 or len(label_arr) == 0:'''
		if len(label_count) == 1 or attr_set_num == 1:
			return new_node

		#calculate the test accuracy up to current node
		a0 = PredictAccuracy(new_node,df_test)

		#get the optimal attribution for new branch
		new_node.attr,div_value = OptAttr_Gini(df_train)#via Gini index

		#get the new branch
		if div_value == 0:
			value_count = ValueCount(df_train[new_node.attr]) 
			for value in value_count: 
				df_v = df_train[ df_train[new_node.attr].isin([value]) ]  # get sub set
				df_v = df_v.drop(new_node.attr, 1)
				#child node
				new_node_child = Node(None,None,{})
				label_arr_child = df_train[df_v.columns[-1]]
				label_count_child = NodeLabel(label_arr_child)
				new_node_child.label = max(label_count_child,key = label_count_child.get)
				new_node.sub_attr[value] = new_node_child

			#calculate to check whether there are more branches
			a1 = PredictAccuracy(new_node,df_test)
			if a1 > a0:#go on branching
				for value in value_count:
					df_v = df_train[df_train[new_node.attr].isin([value])]
					df_v = df_v.drop(new_node.attr,1)
					new_node.sub_attr[value] = TreeGenerate(df_v)
			else:
				new_node.attr = None
				new_node.sub_attr = {}

		else:
			value_l = "<=%.3f" % div_value
			value_r = ">%.3f" % div_value
			df_v_l = df_train[ df_train[new_node.attr] <= div_value ]  # get sub set
			df_v_r = df_train[ df_train[new_node.attr] > div_value ]

			#for child node
			new_node_l = Node(None,None,{})
			new_node_r = Node(None,None,{})
			label_count_l = NodeLabel(df_v_l[df_v_r.columns[-1]])
			label_count_r = NodeLabel(df_v_r[df_v_r.columns[-1]])
			new_node_l.label = max(label_count_l, key=label_count_l.get)
			new_node_r.label = max(label_count_r, key=label_count_r.get)
			new_node.sub_attr[value_l] = new_node_l
			new_node.sub_attr[value_r] = new_node_r

			#calculate to check whether there are more branches
			a1 = PredictAccuracy(new_node, df_test)
			if a1 > a0:  # need branching
				new_node.sub_attr[value_l] = TreeGenerate(df_v_l)
				new_node.sub_attr[value_r] = TreeGenerate(df_v_r)
			else:
				new_node.attr = None
				new_node.sub_attr = {}

	return new_node

'''post-prune a decision tree
@param root: root node
@param df_test: testing set for pruning
@return accuracy score when traversal the tree
'''
def PostPrune(root,df_test):
	#leaf node
	if root.attr == None:
		return PredictAccuracy(root,df_test)
	#calculate the test accuracy on children node
	a1 = 0
	value_count = ValueCount(df_test[root.attr])
	for value in list(value_count):
		df_test_v = df_test[df_test[root.attr].isin([value])]
		if value in root.sub_attr:
			a1_v = PostPrune(root.sub_attr[value],df_test_v)
		else:
			a1_v = PredictAccuracy(root,df_test_v)
		if a1_v == -1:#mean no pruning from this child
			return -1
		else:
			a1 += a1_v*len(df_test_v.index)/len(df_test.index)
	#calculate the test accuracy on this node
	node = Node(None,root.label,{})
	a0 = PredictAccuracy(node,df_test)

	#check if needing pruning
	if a0 >= a1:
		root.attr = None
		root.sub_attr = {}
		return a0
	else:
		return -1

