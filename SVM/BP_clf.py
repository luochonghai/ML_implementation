#coding = <utf-8>
'''BP network for classification on breast_cancer data_set'''

from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError#return the error(%) in the form of list/array
import matplotlib.pyplot as plt

data_set = load_breast_cancer()
X = data_set.data  # feature
feature_names = data_set.feature_names
y = data_set.target  # label
target_names = data_set.target_names

normalized_X = preprocessing.normalize(X)

'''construction of data in pybrain"s formulation '''
ds = ClassificationDataSet(30,1,nb_classes = 2,class_labels = y)
for i in range(len(y)):
	ds.appendLinked(X[i],y[i])
ds.calculateStatistics()#return a class histogram

"""split of training and testing dataset"""
tstdata_temp,trndata_temp = ds.splitWithProportion(0.5)
tstdata = ClassificationDataSet(30,1,nb_classes = 2)#the first parameter inp is used to specify the dimensionality of the input

for n in range(0,tstdata_temp.getLength()):
	tstdata.appendLinked(tstdata_temp.getSample(n)[0],tstdata_temp.getSample(n)[1])

trndata = ClassificationDataSet(30,1,nb_classes = 2)
for n in range(0,trndata_temp.getLength()):
	trndata.appendLinked(trndata_temp.getSample(n)[0],trndata_temp.getSample(n)[1])
trndata._convertToOneOfMany()#convert the target classes to a 1-of-k representation,retaining the old targets as a field class
tstdata._convertToOneOfMany()

n_hidden = 500
bp_nn = buildNetwork(trndata.indim,n_hidden,trndata.outdim,outclass = SoftmaxLayer)
trainer = BackpropTrainer(bp_nn,
							dataset = trndata,
							verbose = True,
							momentum = 0.5,
							learningrate = 0.0001,
							batchlearning = True)
err_train,err_valid = trainer.trainUntilConvergence(maxEpochs = 1000,
													validationProportion = 0.25)
f1 = plt.figure(1)#convergence curve for accumulative BP algorithm process
plt.plot(err_train,'b',err_valid,'r')
plt.title('BP network classification')
plt.ylabel('error rate')
plt.xlabel('epochs')
plt.show()

tst_result = percentError(trainer.testOnClassData(tstdata),tstdata['class'])
print("epoch:%4d"%trainer.totalepochs,"test error:%5.2f%%"%tst_result)
